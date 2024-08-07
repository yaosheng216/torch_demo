from collections import defaultdict

class Solution:
    def numberOfPairs(self, nums1, nums2, k):
        nums1_count = defaultdict(int)
        for num in nums1:
            nums1_count[num] += 1

        quality_pairs_count = 0

        for num in nums2:
            product = num * k
            for value in nums1_count:
                if value % product == 0:
                    quality_pairs_count += nums1_count[value]

        return quality_pairs_count
