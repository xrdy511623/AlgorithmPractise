package Utils


func Max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func Min(a, b int) int {
	if a < b {
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
	if n == 0 {
		return sum
	}
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

func MinAbs(a, b int)int{
	if Abs(a) < Abs(b){
		return a
	}
	return b
}


// ReverseArray 原地反转数组
func ReverseArray(nums []int) []int {
	n := len(nums)
	for i := 0; i < n/2; i++ {
		temp := nums[n-1-i]
		nums[n-1-i] = nums[i]
		nums[i] = temp
	}
	return nums
}


func FindLargestElement(nums []int)(pos int){
	if len(nums) == 0{
		pos = -1
	}
	for i:=0;i<len(nums);i++{
		if nums[i] > nums[pos]{
			pos = i
		}
	}
	return
}