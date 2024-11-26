from collections import defaultdict
from typing import List


def countPairs(nums: List[int]) -> int:
    # 辅助函数，检查两个数字是否可以通过交换一个数的一位变成相等
    def can_become_equal(a, b):
        # 将数字转为字符串并排序
        str_a, str_b = sorted(str(a)), sorted(str(b))
        # 如果排序后的字符串不相等，则直接返回 False
        if str_a != str_b:
            return False
        # 检查两字符串是否完全相同或只有一位不同
        diff_count = sum(1 for x, y in zip(str(a), str(b)) if x != y)
        return diff_count == 0 or diff_count == 2

    # 使用字典记录每个排序后的数字字符串及其出现次数
    sorted_nums_count = defaultdict(int)
    for num in nums:
        sorted_str_num = ''.join(sorted(str(num)))
        sorted_nums_count[sorted_str_num] += 1

    # 计算近似相等数对的数量
    count = 0
    for num in nums:
        sorted_str_num = ''.join(sorted(str(num)))
        for other_sorted_str_num, count_other in sorted_nums_count.items():
            if can_become_equal(num, int(other_sorted_str_num)):
                count += count_other - (sorted_str_num == other_sorted_str_num)

    # 因为每个数对被计算了两次，所以需要除以2
    return count // 2


if __name__ == '__main__':
    print(countPairs([3, 12, 30, 17, 21]))
