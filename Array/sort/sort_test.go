package sort_test

import (
	"algorithm-practise/array/sort"
	"fmt"
	"reflect"
	"testing"
)

func TestBubbleSort(t *testing.T) {
	array := []int{54, 26, 93, 17, 77, 31, 44, 55, 20}
	got := sort.BubbleSort(array)
	want := []int{17, 20, 26, 31, 44, 54, 55, 77, 93}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}

func TestInsertSort(t *testing.T) {
	array := []int{54, 26, 93, 17, 77, 31, 44, 55, 20}
	got := sort.InsertSort(array)
	want := []int{17, 20, 26, 31, 44, 54, 55, 77, 93}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}

func TestQuickSort(t *testing.T) {
	tests := []struct {
		name     string
		input    []int
		expected []int
	}{
		{
			name:     "Empty array",
			input:    []int{},
			expected: []int{},
		},
		{
			name:     "Single element",
			input:    []int{1},
			expected: []int{1},
		},
		{
			name:     "Two elements",
			input:    []int{2, 1},
			expected: []int{1, 2},
		},
		{
			name:     "Already sorted",
			input:    []int{1, 2, 3, 4, 5},
			expected: []int{1, 2, 3, 4, 5},
		},
		{
			name:     "Reverse sorted",
			input:    []int{5, 4, 3, 2, 1},
			expected: []int{1, 2, 3, 4, 5},
		},
		{
			name:     "With duplicates",
			input:    []int{3, 1, 4, 1, 5, 9, 2, 6, 5, 3},
			expected: []int{1, 1, 2, 3, 3, 4, 5, 5, 6, 9},
		},
		{
			name:     "Random order",
			input:    []int{7, 2, 8, 1, 9, 3, 5, 4, 6},
			expected: []int{1, 2, 3, 4, 5, 6, 7, 8, 9},
		},
		{
			name:     "All same elements",
			input:    []int{2, 2, 2, 2, 2},
			expected: []int{2, 2, 2, 2, 2},
		},
		{
			name:     "Negative numbers",
			input:    []int{-3, 5, -1, 0, -2},
			expected: []int{-3, -2, -1, 0, 5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 创建输入副本以避免修改原始输入
			input := make([]int, len(tt.input))
			copy(input, tt.input)

			// 如果数组非空，调用 QuickSort
			if len(input) > 0 {
				sort.QuickSort(input, 0, len(input)-1)
			}

			// 比较实际结果与预期结果
			if !reflect.DeepEqual(input, tt.expected) {
				t.Errorf("QuickSort() = %v, want %v", input, tt.expected)
			}
		})
	}
}

//func TestQuickSort(t *testing.T) {
//	array := []int{54, 26, 93, 17, 77, 31, 44, 55, 20}
//	sort.QuickSort(array, 0, len(array)-1)
//	want := []int{17, 20, 26, 31, 44, 54, 55, 77, 93}
//	if !reflect.DeepEqual(array, want) {
//		t.Errorf("got:%v, expected:%v", array, want)
//	} else {
//		fmt.Println("test pass")
//	}
//}

func TestMergeSort(t *testing.T) {
	array := []int{54, 26, 93, 17, 77, 31, 44, 55, 20}
	got := sort.MergeSort(array)
	want := []int{17, 20, 26, 31, 44, 54, 55, 77, 93}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}

func TestBinarySearch(t *testing.T) {
	array := []int{17, 20, 26, 31, 44, 54, 55, 77, 93}
	got := sort.BinarySearch(array, 26)
	want := 2
	if got != want {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}

func TestBinarySearchUseRecursion(t *testing.T) {
	array := []int{17, 20, 26, 31, 44, 54, 55, 77, 93}
	got := sort.BinarySearchUseRecursion(array, 55)
	want := true
	if got != want {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}

func TestSearchRange(t *testing.T) {
	array := []int{17, 20, 26, 26, 26, 31, 44, 54, 55, 77, 93}
	got := sort.SearchRange(array, 26)
	want := []int{2, 4}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}

func TestSearch(t *testing.T) {
	array := []int{17, 20, 26, 26, 26, 31, 44, 54, 55, 77, 93}
	got := sort.Search(array, 26)
	want := 3
	if got != want {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}

func TestRotate(t *testing.T) {
	nums := []int{1, 2, 3, 4, 5, 6, 7}
	sort.Rotate(nums, 3)
	got := nums
	want := []int{5, 6, 7, 1, 2, 3, 4}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}

func TestFindMinSimple(t *testing.T) {
	nums := []int{3, 4, 5, 1, 2}
	got := sort.FindMin(nums)
	want := 1
	if got != want {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}

func TestSearchRotateArray(t *testing.T) {
	array := []int{15, 16, 19, 20, 25, 1, 3, 4, 5, 7, 10, 14}
	got := sort.SearchRotateArray(array, 7)
	want := 9
	if got != want {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}

func TestReversePairs(t *testing.T) {
	record := []int{9, 7, 5, 4, 6}
	got := sort.ReversePairs(record)
	want := 8
	if got != want {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}
