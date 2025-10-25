class RingBuffer:
    def __init__(self, capacity: int):
        """
        创建一个新的环形缓冲区
        :param capacity: 缓冲区容量
        """
        self.size = capacity
        self.buf = [0.0] * capacity
        self.idx = 0

    def push(self, val: float):
        """
        向缓冲区压入一个元素
        """
        self.buf[self.idx] = val
        self.idx = (self.idx + 1) % self.size

    def get_buf(self):
        """
        获取旋转对齐后的缓冲区内容
        :return: list[float]
        """
        return [self.buf[(self.idx + i) % self.size] for i in range(self.size)]

    def print_buf(self, tag: str = "RingBuffer"):
        """
        打印缓冲区内容
        """
        print(f"[{tag}] ", end="")
        for val in self.buf:
            print(f"{val:.6f} ", end="")
        print()


# 使用示例
if __name__ == "__main__":
    rb = RingBuffer(5)  # 创建容量为 5 的环形缓冲区
    for i in range(7):
        rb.push(float(i))
        rb.print_buf("example")

    print("旋转后的缓冲区内容:", rb.get_buf())
