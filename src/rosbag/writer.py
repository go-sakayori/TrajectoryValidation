class RosbagWriter:
    def __init__(self, bag_file):
        import rosbag
        self.bag_file = bag_file
        self.bag = rosbag.Bag(bag_file, 'w')

    def write_bag(self, topic, data, timestamp):
        self.bag.write(topic, data, timestamp)

    def close(self):
        self.bag.close()