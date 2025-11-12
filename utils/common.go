package utils

func Max(a, b int) int {
	if a >= b {
		return a
	}
	return b
}

func Min(a, b int) int {
	if a <= b {
		return a
	}
	return b
}

func MaxValueOfArray(array []int) int {
	max := array[0]
	for i := 1; i < len(array); i++ {
		if array[i] > max {
			max = array[i]
		}
	}
	return max
}

func MinValueOfArray(array []int) int {
	min := array[0]
	for i := 1; i < len(array); i++ {
		if array[i] < min {
			min = array[i]
		}
	}
	return min
}

func SumOfArray(array []int) int {
	sum := 0
	n := len(array)
	for i := 0; i < n; i++ {
		sum += array[i]
	}
	return sum
}

func Abs(x int) int {
	if x < 0 {
		return -1 * x
	}
	return x
}

func MinAbs(a, b int) int {
	if Abs(a) < Abs(b) {
		return Abs(a)
	}
	return Abs(b)
}

// ReverseArray 原地反转数组
func ReverseArray(nums []int) []int {
	for i, n := 0, len(nums); i < n/2; i++ {
		nums[i], nums[n-1-i] = nums[n-1-i], nums[i]
	}
	return nums
}

func FindLargestElement(nums []int) int {
	pos := 0
	for i, n := 0, len(nums); i < n; i++ {
		if nums[i] > nums[pos] {
			pos = i
		}
	}
	return pos
}

func ReverseString(s []byte) string {
	for i, n := 0, len(s); i < n/2; i++ {
		s[i], s[n-1-i] = s[n-1-i], s[i]
	}
	return string(s)
}

func CheckAlphaNumeric(char byte) bool {
	return char >= '0' && char <= '9' || char >= 'a' && char <= 'z' || char >= 'A' && char <= 'Z'
}

// CompareDesc 自定义降序规则
func CompareDesc(x, y string) bool {
	return x+y > y+x
}

// CompareAsc 自定义升序规则
func CompareAsc(x, y string) bool {
	return x+y < y+x
}

// ReverseStr 返回字符串 s 的反转结果
func ReverseStr(s string) string {
	// 使用 strings.Builder 也可以，但这里用 []byte 直接操作
	b := []byte(s)
	for i, n := 0, len(b); i < n/2; i++ {
		b[i], b[n-1-i] = b[n-1-i], b[i]
	}
	return string(b)
}

// CheckPalindrome 检查字符串 s 是否为回文
func CheckPalindrome(s string) bool {
	i, j := 0, len(s)-1
	for i < j {
		if s[i] != s[j] {
			return false
		}
		i++
		j--
	}
	return true
}
