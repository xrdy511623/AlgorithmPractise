package Utils

func Max(a, b int) int {
	if a > b {
		return a
	} else {
		return b
	}
}

func Min(a, b int) int {
	if a < b{
		return a
	} else{
		return b
	}
}

func MaxValueOfArray(array []int)int{
	max := array[0]
	for i:=1;i<len(array);i++{
		if array[i] > max{
			max = array[i]
		}
	}
	return max
}
