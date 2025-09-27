#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2

class TestPC2Pub(Node):
    def __init__(self):
        super().__init__('test_pc2_pub')
        self.pub = self.create_publisher(PointCloud2, 'input_points', 10)
        self.timer = self.create_timer(1.0, self.timer_cb)
        self.get_logger().info('TestPC2Pub started, publishing to "input_points"')

    def timer_cb(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        # fields: x,y,z,int32 id
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='id', offset=12, datatype=PointField.INT32, count=1),
        ]
        
        points = [
            (10.0, 0.0, 0, 0),
            (12.0, 0.0, 0.5, 0),
            (12.0, 1.0, 0, 0),
            (10.0, 1.0, 0.5, 0),
            (5.0, 5.0, 0, 1),
            (5.5, 5.0, 0.2, 1),
            (5.5, 5.5, 0, 1),
            (5.0, 5.5, 0.2, 1),
            (8.0, 2.0, 0.5, 2),
            (8.1, 2.1, 3.0, 2),
            (8.2, 2.2, 3.5, 2),
        ]



        cloud = point_cloud2.create_cloud(header, fields, points)
        self.pub.publish(cloud)
        self.get_logger().info(f'Published test cloud with {len(points)} points (clusters 0 and 1)')

def main(args=None):
    rclpy.init(args=args)
    node = TestPC2Pub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
