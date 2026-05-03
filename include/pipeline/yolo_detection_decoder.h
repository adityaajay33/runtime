#pragma once

#include "core/detection.h"
#include "core/frame.h"
#include "core/tensor.h"
#include "pipeline/image_preprocessor.h"

#include <string>
#include <vector>

namespace ptk
{

    struct YoloDetectionDecoderConfig
    {
        float confidence_threshold = 0.35F;
        float iou_threshold = 0.45F;

        int target_class_id = 15;
        std::string target_class_name = "cat";
    };

    class YoloDetectionDecoder
    {
    public:
        explicit YoloDetectionDecoder(YoloDetectionDecoderConfig config);

        bool run(
            const TensorView &model_output,
            const Frame &frame,
            const PreprocessInfo &preprocess_info,
            PerceptionResult &result);

    private:
        struct Candidate
        {
            BoundingBox2D bbox;
            float confidence = 0.0F;
            int class_id = -1;
        };

        // Column ordering for the [1, N, 6] pre-decoded format
        enum class N6Format { ConfThenClass, ClassThenConf };

        // [1, N, 6]: already NMS-filtered rows of [x1,y1,x2,y2,conf,cls] or swapped
        bool decode_prefiltered_n6(
            const float *data,
            const TensorShape &shape,
            const PreprocessInfo &preprocess_info,
            std::vector<Candidate> &candidates);

        // Raw YOLO [1, C, N] where C < N (channels-first, e.g. [1, 84, 8400])
        bool decode_layout_channels_first(
            const float *data,
            const TensorShape &shape,
            const PreprocessInfo &preprocess_info,
            std::vector<Candidate> &candidates) const;

        // Raw YOLO [1, N, C] where N > C (boxes-first, e.g. [1, 8400, 84])
        bool decode_layout_boxes_first(
            const float *data,
            const TensorShape &shape,
            const PreprocessInfo &preprocess_info,
            std::vector<Candidate> &candidates) const;

        // Map cx,cy,w,h (model/letterbox space) → BoundingBox2D in original image space
        BoundingBox2D map_box_to_original_image(
            float center_x,
            float center_y,
            float box_w,
            float box_h,
            const PreprocessInfo &preprocess_info) const;

        // Map x1,y1,x2,y2 (model/letterbox space) → BoundingBox2D in original image space
        BoundingBox2D map_xyxy_to_original_image(
            float x1, float y1, float x2, float y2,
            const PreprocessInfo &preprocess_info) const;

        // Returns true if max observed x/y coords fit within model dimensions
        // (i.e. coords are in model/letterbox space and need unmapping)
        bool coords_are_in_model_space(
            float max_x, float max_y,
            const PreprocessInfo &preprocess_info) const;

        std::vector<Candidate> apply_nms(const std::vector<Candidate> &candidates) const;

        float intersection_over_union(const BoundingBox2D &a, const BoundingBox2D &b) const;

        YoloDetectionDecoderConfig config_;

        // State persisted across frames for the [1, N, 6] path
        bool debug_logged_ = false;
        bool n6_format_detected_ = false;
        N6Format n6_format_ = N6Format::ConfThenClass;
    };

} // namespace ptk
