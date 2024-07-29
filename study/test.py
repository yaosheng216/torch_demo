
def clearStars(s) -> str:
    stack = []
    for ch in s:
        if ch == '*':
            # 找到 '*' 左边距离 '*' 最近的一个字典序最小的字符并删除它
            min_char = min(stack)
            # 删除最左边的最小字符
            stack.remove(min_char)
        else:
            stack.append(ch)
    return ''.join(stack)


if __name__ == "__main__":
    print(clearStars("aaba*"))
