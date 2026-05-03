#include "pipeline/yolo_detection_decoder.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>

namespace ptk
{

    YoloDetectionDecoder::YoloDetectionDecoder(YoloDetectionDecoderConfig config)
        : config_(std::move(config)) {}

    bool YoloDetectionDecoder::run(
        const TensorView &model_output,
        const Frame &frame,
        const PreprocessInfo &preprocess_info,
        PerceptionResult &result)
    {
        if (!model_output.valid() || !frame.valid())
        {
            return false;
        }

        if (model_output.dtype != DataType::Float32)
        {
            return false;
        }

        const auto &dims = model_output.shape.dims;

        if (dims.size() != 3)
        {
            std::cerr << "[decoder] expected rank-3 output, got rank " << dims.size() << "\n";
            return false;
        }

        const auto *data = model_output.data_as<float>();

        if (data == nullptr)
        {
            return false;
        }

        std::vector<Candidate> candidates;
        bool apply_nms_flag = true;

        if (dims[2] == 6)
        {
            // Pre-decoded format: [1, N, 6] — model already applied NMS
            if (!decode_prefiltered_n6(data, model_output.shape, preprocess_info, candidates))
            {
                return false;
            }
            apply_nms_flag = false;
        }
        else
        {
            // Raw YOLO output: distinguish [1, C, N] vs [1, N, C]
            // channels_first = true  → [1, C, N], C is small (e.g. 84), N is large (e.g. 8400)
            // channels_first = false → [1, N, C], boxes dimension first
            const bool channels_first = dims[1] < dims[2];

            if (channels_first)
            {
                if (!decode_layout_channels_first(data, model_output.shape, preprocess_info, candidates))
                {
                    return false;
                }
            }
            else
            {
                if (!decode_layout_boxes_first(data, model_output.shape, preprocess_info, candidates))
                {
                    return false;
                }
            }
        }

        const std::vector<Candidate> kept =
            apply_nms_flag ? apply_nms(candidates) : std::move(candidates);

        result.detections.clear();
        result.timestamp = frame.timestamp;
        result.frame_sequence = frame.sequence;
        result.frame_id = frame.frame_id;
        result.source_id = frame.source_id;

        for (const auto &candidate : kept)
        {
            Detection2D detection;
            detection.class_id = candidate.class_id;
            detection.class_name = config_.target_class_name;
            detection.confidence = candidate.confidence;
            detection.bbox = candidate.bbox;
            detection.timestamp = frame.timestamp;
            detection.frame_sequence = frame.sequence;
            detection.frame_id = frame.frame_id;
            detection.source_id = frame.source_id;

            if (detection.valid())
            {
                result.detections.push_back(detection);
            }
        }

        return true;
    }

    // ---------------------------------------------------------------------------
    // Pre-decoded [1, N, 6] path
    // ---------------------------------------------------------------------------

    bool YoloDetectionDecoder::decode_prefiltered_n6(
        const float *data,
        const TensorShape &shape,
        const PreprocessInfo &preprocess_info,
        std::vector<Candidate> &candidates)
    {
        const int num_rows = static_cast<int>(shape.dims[1]);

        if (num_rows <= 0)
        {
            return true;
        }

        // 1. Determine coordinate space: scan all rows for the largest x2/y2.
        //    If they fit within model dimensions the coords are in model/letterbox
        //    space and need to be unmapped. If they exceed model dimensions the
        //    model already output original-image coordinates.
        float max_x = 0.0F;
        float max_y = 0.0F;

        for (int i = 0; i < num_rows; ++i)
        {
            const float *row = data + i * 6;
            max_x = std::max(max_x, row[2]);
            max_y = std::max(max_y, row[3]);
        }

        const bool in_model_space = coords_are_in_model_space(max_x, max_y, preprocess_info);

        // Helper: map a single xyxy box to original-image BoundingBox2D.
        auto make_bbox = [&](float x1, float y1, float x2, float y2) -> BoundingBox2D
        {
            if (in_model_space)
            {
                return map_xyxy_to_original_image(x1, y1, x2, y2, preprocess_info);
            }

            const float ow = static_cast<float>(preprocess_info.original_width);
            const float oh = static_cast<float>(preprocess_info.original_height);
            const float cx1 = std::clamp(x1, 0.0F, ow);
            const float cy1 = std::clamp(y1, 0.0F, oh);
            const float cx2 = std::clamp(x2, 0.0F, ow);
            const float cy2 = std::clamp(y2, 0.0F, oh);
            return BoundingBox2D{cx1, cy1, cx2 - cx1, cy2 - cy1};
        };

        // 2. Auto-detect column format once (persisted across frames).
        //    Look at the first row with a valid bounding box.
        //    If col-4 is integer-like (class id) and col-5 is in (0,1] → ClassThenConf.
        //    Otherwise default to ConfThenClass.
        if (!n6_format_detected_)
        {
            n6_format_ = N6Format::ConfThenClass;

            for (int i = 0; i < num_rows; ++i)
            {
                const float *row = data + i * 6;

                if (row[2] <= row[0] || row[3] <= row[1])
                {
                    continue;
                }

                const float v4 = row[4];
                const float v5 = row[5];

                const bool v4_integer = (v4 >= 0.0F && (v4 - std::floor(v4)) < 0.05F);
                const bool v5_conf    = (v5 > 0.0F && v5 <= 1.0F);

                if (v4_integer && v5_conf)
                {
                    n6_format_ = N6Format::ClassThenConf;
                }

                break;
            }

            n6_format_detected_ = true;
        }

        const bool is_conf_then_class = (n6_format_ == N6Format::ConfThenClass);
        const int  conf_col = is_conf_then_class ? 4 : 5;
        const int  cls_col  = is_conf_then_class ? 5 : 4;

        // 3. First-frame debug header.
        const bool first_frame = !debug_logged_;

        if (first_frame)
        {
            debug_logged_ = true;

            std::cout << "[decoder] shape=["
                      << shape.dims[0] << ", " << num_rows << ", 6]\n";
            std::cout << "[decoder] coord space: "
                      << (in_model_space
                             ? "model-space (will unmap via pad + scale)"
                             : "original-image-space (no unmap needed)")
                      << "\n";
            std::cout << "[decoder] column order: "
                      << (is_conf_then_class
                             ? "[x1, y1, x2, y2, confidence, class_id]"
                             : "[x1, y1, x2, y2, class_id, confidence]")
                      << "\n";
            std::cout << "[decoder] rows scanned: " << num_rows << "\n";
            std::cout << "[decoder] first 5 rows (raw):\n";

            for (int i = 0; i < std::min(5, num_rows); ++i)
            {
                const float *r = data + i * 6;
                std::cout << "  [" << i << "]"
                          << "  x1=" << r[0]
                          << "  y1=" << r[1]
                          << "  x2=" << r[2]
                          << "  y2=" << r[3]
                          << "  [4]=" << r[4]
                          << "  [5]=" << r[5] << "\n";
            }
        }

        // 4. Collect candidates with the detected (primary) format.
        candidates.reserve(static_cast<std::size_t>(num_rows));

        for (int i = 0; i < num_rows; ++i)
        {
            const float *row  = data + i * 6;
            const float  x1   = row[0];
            const float  y1   = row[1];
            const float  x2   = row[2];
            const float  y2   = row[3];
            const float  conf = row[conf_col];
            const int    cls  = static_cast<int>(std::round(row[cls_col]));

            if (x2 <= x1 || y2 <= y1)                continue;
            if (cls  != config_.target_class_id)      continue;
            if (conf <  config_.confidence_threshold)  continue;

            const BoundingBox2D bbox = make_bbox(x1, y1, x2, y2);

            if (!bbox.valid())
            {
                continue;
            }

            candidates.push_back(Candidate{bbox, conf, cls});
        }

        if (first_frame)
        {
            std::cout << "[decoder] candidates after class/conf filter: "
                      << candidates.size() << "\n";
        }

        // 5. Fallback: if primary format produced nothing, probe the swapped layout.
        //    If it has matches, switch permanently and re-decode.
        if (candidates.empty())
        {
            const int swapped_conf = cls_col;
            const int swapped_cls  = conf_col;

            bool swapped_has_match = false;

            for (int i = 0; i < num_rows; ++i)
            {
                const float *row = data + i * 6;

                if (row[2] <= row[0] || row[3] <= row[1])
                {
                    continue;
                }

                const float conf2 = row[swapped_conf];
                const int   cls2  = static_cast<int>(std::round(row[swapped_cls]));

                if (cls2 == config_.target_class_id && conf2 >= config_.confidence_threshold)
                {
                    swapped_has_match = true;
                    break;
                }
            }

            if (swapped_has_match)
            {
                n6_format_ = is_conf_then_class ? N6Format::ClassThenConf
                                                : N6Format::ConfThenClass;

                const bool new_ctc   = (n6_format_ == N6Format::ConfThenClass);
                const int  new_conf  = new_ctc ? 4 : 5;
                const int  new_cls   = new_ctc ? 5 : 4;

                std::cout << "[decoder] primary format had 0 candidates; "
                             "switching permanently to "
                          << (new_ctc ? "[x1,y1,x2,y2,confidence,class_id]"
                                      : "[x1,y1,x2,y2,class_id,confidence]")
                          << "\n";

                candidates.clear();

                for (int i = 0; i < num_rows; ++i)
                {
                    const float *row  = data + i * 6;
                    const float  x1   = row[0];
                    const float  y1   = row[1];
                    const float  x2   = row[2];
                    const float  y2   = row[3];
                    const float  conf = row[new_conf];
                    const int    cls  = static_cast<int>(std::round(row[new_cls]));

                    if (x2 <= x1 || y2 <= y1)               continue;
                    if (cls  != config_.target_class_id)     continue;
                    if (conf <  config_.confidence_threshold) continue;

                    const BoundingBox2D bbox = make_bbox(x1, y1, x2, y2);

                    if (!bbox.valid())
                    {
                        continue;
                    }

                    candidates.push_back(Candidate{bbox, conf, cls});
                }
            }
        }

        return true;
    }

    // ---------------------------------------------------------------------------
    // Raw YOLO paths (unchanged logic, kept for [1, 84, N] and [1, N, 84] etc.)
    // ---------------------------------------------------------------------------

    bool YoloDetectionDecoder::decode_layout_channels_first(
        const float *data,
        const TensorShape &shape,
        const PreprocessInfo &preprocess_info,
        std::vector<Candidate> &candidates) const
    {
        const int channels = static_cast<int>(shape.dims[1]);
        const int boxes    = static_cast<int>(shape.dims[2]);

        if (channels < 5 || config_.target_class_id < 0)
        {
            return false;
        }

        const int class_channel = 4 + config_.target_class_id;

        if (class_channel >= channels)
        {
            std::cerr << "[decoder] channels_first: class channel "
                      << class_channel << " >= total channels " << channels << "\n";
            return false;
        }

        for (int i = 0; i < boxes; ++i)
        {
            const float center_x = data[0 * boxes + i];
            const float center_y = data[1 * boxes + i];
            const float width    = data[2 * boxes + i];
            const float height   = data[3 * boxes + i];
            const float conf     = data[class_channel * boxes + i];

            if (conf < config_.confidence_threshold)
            {
                continue;
            }

            Candidate candidate;
            candidate.bbox = map_box_to_original_image(
                center_x, center_y, width, height, preprocess_info);
            candidate.confidence = conf;
            candidate.class_id   = config_.target_class_id;

            if (candidate.bbox.valid())
            {
                candidates.push_back(candidate);
            }
        }

        return true;
    }

    bool YoloDetectionDecoder::decode_layout_boxes_first(
        const float *data,
        const TensorShape &shape,
        const PreprocessInfo &preprocess_info,
        std::vector<Candidate> &candidates) const
    {
        const int boxes    = static_cast<int>(shape.dims[1]);
        const int channels = static_cast<int>(shape.dims[2]);

        if (channels < 5 || config_.target_class_id < 0)
        {
            return false;
        }

        const int class_index = 4 + config_.target_class_id;

        if (class_index >= channels)
        {
            std::cerr << "[decoder] boxes_first: class index "
                      << class_index << " >= row width " << channels << "\n";
            return false;
        }

        for (int i = 0; i < boxes; ++i)
        {
            const float *prediction = data + static_cast<std::size_t>(i) * channels;

            const float center_x = prediction[0];
            const float center_y = prediction[1];
            const float width    = prediction[2];
            const float height   = prediction[3];
            const float conf     = prediction[class_index];

            if (conf < config_.confidence_threshold)
            {
                continue;
            }

            Candidate candidate;
            candidate.bbox = map_box_to_original_image(
                center_x, center_y, width, height, preprocess_info);
            candidate.confidence = conf;
            candidate.class_id   = config_.target_class_id;

            if (candidate.bbox.valid())
            {
                candidates.push_back(candidate);
            }
        }

        return true;
    }

    // ---------------------------------------------------------------------------
    // Coordinate mapping helpers
    // ---------------------------------------------------------------------------

    BoundingBox2D YoloDetectionDecoder::map_box_to_original_image(
        float center_x,
        float center_y,
        float box_w,
        float box_h,
        const PreprocessInfo &preprocess_info) const
    {
        const float x1 = center_x - box_w * 0.5F;
        const float y1 = center_y - box_h * 0.5F;
        const float x2 = center_x + box_w * 0.5F;
        const float y2 = center_y + box_h * 0.5F;

        return map_xyxy_to_original_image(x1, y1, x2, y2, preprocess_info);
    }

    BoundingBox2D YoloDetectionDecoder::map_xyxy_to_original_image(
        float x1, float y1, float x2, float y2,
        const PreprocessInfo &preprocess_info) const
    {
        const float px1 = (x1 - static_cast<float>(preprocess_info.pad_x)) /
                          preprocess_info.scale;
        const float py1 = (y1 - static_cast<float>(preprocess_info.pad_y)) /
                          preprocess_info.scale;
        const float px2 = (x2 - static_cast<float>(preprocess_info.pad_x)) /
                          preprocess_info.scale;
        const float py2 = (y2 - static_cast<float>(preprocess_info.pad_y)) /
                          preprocess_info.scale;

        const float ow = static_cast<float>(preprocess_info.original_width);
        const float oh = static_cast<float>(preprocess_info.original_height);

        const float cx1 = std::clamp(px1, 0.0F, ow);
        const float cy1 = std::clamp(py1, 0.0F, oh);
        const float cx2 = std::clamp(px2, 0.0F, ow);
        const float cy2 = std::clamp(py2, 0.0F, oh);

        return BoundingBox2D{cx1, cy1, cx2 - cx1, cy2 - cy1};
    }

    bool YoloDetectionDecoder::coords_are_in_model_space(
        float max_x, float max_y,
        const PreprocessInfo &preprocess_info) const
    {
        const float mw = static_cast<float>(preprocess_info.model_width);
        const float mh = static_cast<float>(preprocess_info.model_height);

        // If any coord exceeds the model resolution the model output is already
        // in original-image space.  Allow a small margin for fp rounding.
        if (max_x > mw + 2.0F || max_y > mh + 2.0F)
        {
            return false;
        }

        // Coords fit within model bounds → assume model/letterbox space.
        return true;
    }

    // ---------------------------------------------------------------------------
    // NMS
    // ---------------------------------------------------------------------------

    std::vector<YoloDetectionDecoder::Candidate> YoloDetectionDecoder::apply_nms(
        const std::vector<Candidate> &candidates) const
    {
        std::vector<Candidate> sorted = candidates;

        std::sort(
            sorted.begin(),
            sorted.end(),
            [](const Candidate &a, const Candidate &b)
            {
                return a.confidence > b.confidence;
            });

        std::vector<Candidate> kept;

        for (const auto &candidate : sorted)
        {
            bool should_keep = true;

            for (const auto &kept_candidate : kept)
            {
                if (intersection_over_union(candidate.bbox, kept_candidate.bbox) >
                    config_.iou_threshold)
                {
                    should_keep = false;
                    break;
                }
            }

            if (should_keep)
            {
                kept.push_back(candidate);
            }
        }

        return kept;
    }

    float YoloDetectionDecoder::intersection_over_union(
        const BoundingBox2D &a,
        const BoundingBox2D &b) const
    {
        const float ax1 = a.x;
        const float ay1 = a.y;
        const float ax2 = a.x + a.width;
        const float ay2 = a.y + a.height;

        const float bx1 = b.x;
        const float by1 = b.y;
        const float bx2 = b.x + b.width;
        const float by2 = b.y + b.height;

        const float ix1 = std::max(ax1, bx1);
        const float iy1 = std::max(ay1, by1);
        const float ix2 = std::min(ax2, bx2);
        const float iy2 = std::min(ay2, by2);

        const float intersection_w    = std::max(0.0F, ix2 - ix1);
        const float intersection_h    = std::max(0.0F, iy2 - iy1);
        const float intersection_area = intersection_w * intersection_h;

        const float area_a     = std::max(0.0F, a.width) * std::max(0.0F, a.height);
        const float area_b     = std::max(0.0F, b.width) * std::max(0.0F, b.height);
        const float union_area = area_a + area_b - intersection_area;

        if (union_area <= 0.0F)
        {
            return 0.0F;
        }

        return intersection_area / union_area;
    }

} // namespace ptk
