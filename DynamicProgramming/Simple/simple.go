package Simple


/*
1.0 斐波拉契数
斐波那契数，通常用F(n) 表示，形成的序列称为斐波那契数列 。该数列由0和1开始，后面的每一项数字都是前面两项数字的和。也就是：

F(0)= 0，F(1)= 1
F(n) = F(n-1) + F(n-2)，其中 n > 1
给你n ，请计算 F(n) 。
 */

// Fib 递推公式题目已经给出:F(n) = F(n-1) + F(n-2). 时间复杂度O(N), 空间复杂度O(N)
func Fib(n int)int{
	if n <= 1{
		return n
	}
	dp := make([]int, n+1, n+1)
	dp[0] = 0
	dp[1] = 1
	for i:=2;i<=n;i++{
		dp[i] = dp[i-1]+dp[i-2]
	}
	return dp[n]
}

// 当然，上面的方法还有优化的空间，其实我们只需要维护两个数值就可以了，不需要记录整个序列
// 时间复杂度O(N), 空间复杂度O(1)

func FibSimple(n int)int{
	if n <= 1{
		return n
	}
	dp := make([]int, 2, 2)
	dp[0] = 0
	dp[1] = 1
	for i:=2;i<=n;i++{
		sum := dp[0]+dp[1]
		dp[0] = dp[1]
		dp[1] = sum
	}
	return dp[1]
}

// FibUseRecursion 当然，本题还可以用递归解决，但是会有大量重复计算，时间复杂度O(2^N), 空间复杂度O(N)
func FibUseRecursion(n int)int{
	if n < 2{
		return n
	}
	return FibUseRecursion(n-1)+FibUseRecursion(n-2)
}

// FibUseRecursionAndCache 要降低时间复杂度，需要使用缓存, 时间复杂度O(N), 空间复杂度O(N)
func FibUseRecursionAndCache(n int)int{
	cache := make(map[int]int, 0)
	var helper func(int)int
	helper = func(n int)int{
		if v, ok := cache[n];ok{
			return v
		}
		if n < 2 {
			cache[n] = n
			return n
		}
		cache[n] = helper(n-1)+helper(n-2)
		return cache[n]
	}
	return helper(n)
}
