package sort

import (
	"fmt"
	"learngo/sort"
	"reflect"
	"testing"
)

func TestBubbleSort(t *testing.T) {
	array := []int{54, 26, 93, 17, 77, 31, 44, 55, 20}
	got := BubbleSort(array)
	want := []int{17, 20, 26, 31, 44, 54, 55, 77, 93}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}

func TestInsertSort(t *testing.T) {
	array := []int{54, 26, 93, 17, 77, 31, 44, 55, 20}
	got := InsertSort(array)
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
	got := MergeSort(array)
	want := []int{17, 20, 26, 31, 44, 54, 55, 77, 93}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}

func TestBinarySearch(t *testing.T) {
	array := []int{17, 20, 26, 31, 44, 54, 55, 77, 93}
	got := BinarySearch(array, 26)
	want := 2
	if got != want {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}

func TestBinarySearchUseRecursion(t *testing.T) {
	array := []int{17, 20, 26, 31, 44, 54, 55, 77, 93}
	got := BinarySearchUseRecursion(array, 55)
	want := true
	if got != want {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}

func TestSearchRange(t *testing.T) {
	array := []int{17, 20, 26, 26, 26, 31, 44, 54, 55, 77, 93}
	got := SearchRange(array, 26)
	want := []int{2, 4}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}

func TestSearch(t *testing.T) {
	array := []int{17, 20, 26, 26, 26, 31, 44, 54, 55, 77, 93}
	got := Search(array, 26)
	want := 3
	if got != want {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}

func TestRotate(t *testing.T) {
	nums := []int{1, 2, 3, 4, 5, 6, 7}
	Rotate(nums, 3)
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
	got := FindMin(nums)
	want := 1
	if got != want {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}

func TestSearchRotateArray(t *testing.T) {
	array := []int{15, 16, 19, 20, 25, 1, 3, 4, 5, 7, 10, 14}
	got := SearchRotateArray(array, 7)
	want := 9
	if got != want {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}

func TestReversePairs(t *testing.T) {
	record := []int{9, 7, 5, 4, 6}
	got := ReversePairs(record)
	want := 8
	if got != want {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}

// 测试用例结构
type testCase struct {
	name     string
	users    []*userOnLine
	expected int64
}

func TestGetMaxOnlineTime(t *testing.T) {
	testCases := []testCase{
		{
			name: "Basic case",
			users: []*userOnLine{
				{uid: 1, loginTime: 100, logoutTime: 300},
				{uid: 2, loginTime: 200, logoutTime: 400},
				{uid: 3, loginTime: 250, logoutTime: 350},
				{uid: 4, loginTime: 150, logoutTime: 280},
			},
			expected: 250,
		},
		{
			name: "All users log in at the same time",
			users: []*userOnLine{
				{uid: 1, loginTime: 100, logoutTime: 200},
				{uid: 2, loginTime: 100, logoutTime: 300},
				{uid: 3, loginTime: 100, logoutTime: 400},
			},
			expected: 100,
		},
		{
			name: "Users log in and out sequentially",
			users: []*userOnLine{
				{uid: 1, loginTime: 100, logoutTime: 150},
				{uid: 2, loginTime: 150, logoutTime: 200},
				{uid: 3, loginTime: 200, logoutTime: 250},
			},
			expected: 100, // 第一个用户登录时刻是最大在线人数首次出现的时刻
		},
		{
			name: "Only one user",
			users: []*userOnLine{
				{uid: 1, loginTime: 500, logoutTime: 800},
			},
			expected: 500,
		},
		{
			name: "Users log out at the same time",
			users: []*userOnLine{
				{uid: 1, loginTime: 100, logoutTime: 300},
				{uid: 2, loginTime: 200, logoutTime: 300},
				{uid: 3, loginTime: 250, logoutTime: 300},
			},
			expected: 250,
		},
		{
			name: "Users stay online the whole day",
			users: []*userOnLine{
				{uid: 1, loginTime: 0, logoutTime: 86400},
				{uid: 2, loginTime: 0, logoutTime: 86400},
				{uid: 3, loginTime: 0, logoutTime: 86400},
			},
			expected: 0,
		},
		{
			name: "Users log in at different times",
			users: []*userOnLine{
				{uid: 1, loginTime: 100, logoutTime: 500},
				{uid: 2, loginTime: 200, logoutTime: 600},
				{uid: 3, loginTime: 300, logoutTime: 700},
				{uid: 4, loginTime: 400, logoutTime: 800},
				{uid: 5, loginTime: 500, logoutTime: 900},
			},
			expected: 400,
		},
		{
			name:     "No users",
			users:    []*userOnLine{},
			expected: 0, // 没有用户，默认返回 0
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := getMaxOnlineTime(tc.users)
			if result != tc.expected {
				t.Errorf("Expected %d, but got %d", tc.expected, result)
			}
		})
	}
}
