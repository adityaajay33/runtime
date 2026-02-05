"""
Launch file for PTK perception pipeline with composable nodes.
Loads camera, preprocessor, and inference into a single container.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    #declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='Path to the ONNX model file'
    )
    
    device_index_arg = DeclareLaunchArgument(
        'device_index',
        default_value='0',
        description='Camera device index'
    )
    
    target_height_arg = DeclareLaunchArgument(
        'target_height',
        default_value='224',
        description='Preprocessor target height'
    )
    
    target_width_arg = DeclareLaunchArgument(
        'target_width',
        default_value='224',
        description='Preprocessor target width'
    )
    
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.5',
        description='Detection confidence threshold'
    )
    
    #create composable node container with all pipeline components
    container = ComposableNodeContainer(
        name='perception_container',
        namespace='ptk',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='ptk',
                plugin='ptk::sensors::MacCamera',
                name='camera',
                parameters=[{
                    'device_index': LaunchConfiguration('device_index'),
                }],
            ),
            ComposableNode(
                package='ptk',
                plugin='ptk::Preprocessor',
                name='preprocessor',
                parameters=[{
                    'target_height': LaunchConfiguration('target_height'),
                    'target_width': LaunchConfiguration('target_width'),
                    'normalize': True,
                }],
            ),
            ComposableNode(
                package='ptk',
                plugin='ptk::components::InferenceNode',
                name='inference',
                parameters=[{
                    'model_path': LaunchConfiguration('model_path'),
                    'backend': 'onnx',
                    'task_type': 'detection',
                    'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                }],
            ),
            ComposableNode(
                package='ptk',
                plugin='ptk::components::FramePublisherBridge',
                name='frame_publisher',
                parameters=[{
                    'topic_name': 'ptk/camera/image',
                    'frame_id': 'camera',
                }],
            ),
        ],
        output='both',
    )
    
    return LaunchDescription([
        model_path_arg,
        device_index_arg,
        target_height_arg,
        target_width_arg,
        confidence_threshold_arg,
        container,
    ])
