class Solution:
    def doesAliceWin(self, s: str) -> bool:
        n = len(s)
        # dp[i][j] 表示 s[i:j+1] 是否是小红的胜利状态
        dp = [[False] * n for _ in range(n)]

        # 计算每个子字符串的元音个数
        vowel_count = [[0] * n for _ in range(n)]
        vowels = set('aeiou')

        for i in range(n):
            vowel_count[i][i] = 1 if s[i] in vowels else 0
            for j in range(i + 1, n):
                vowel_count[i][j] = vowel_count[i][j - 1] + (1 if s[j] in vowels else 0)

        # 从短的子字符串开始填 dp 表
        for length in range(1, n + 1):  # 子字符串长度
            for i in range(n - length + 1):
                j = i + length - 1
                if vowel_count[i][j] % 2 == 1:  # 奇数个元音
                    dp[i][j] = False  # 小红不能直接获胜，需查看是否有有效子字符串可以移除
                    for k in range(i, j + 1):
                        for l in range(k, j + 1):
                            if vowel_count[k][l] % 2 == 1:
                                left_lost = (i <= k - 1 and not dp[i][k - 1]) or (i > k - 1)
                                right_lost = (l + 1 <= j and not dp[l + 1][j]) or (l + 1 > j)
                                if left_lost and right_lost:
                                    dp[i][j] = True
                                    break
                        if dp[i][j]:
                            break
                else:  # 偶数个元音
                    dp[i][j] = False  # 小明不能直接获胜，需查看是否有有效子字符串可以移除
                    for k in range(i, j + 1):
                        for l in range(k, j + 1):
                            if vowel_count[k][l] % 2 == 0:
                                left_lost = (i <= k - 1 and not dp[i][k - 1]) or (i > k - 1)
                                right_lost = (l + 1 <= j and not dp[l + 1][j]) or (l + 1 > j)
                                if left_lost and right_lost:
                                    dp[i][j] = True
                                    break
                        if dp[i][j]:
                            break

        # 返回整个字符串的胜利状态
        return dp[0][n - 1]
