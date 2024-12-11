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
	array := []int{54, 26, 93, 17, 77, 31, 44, 55, 20}
	sort.QuickSort(array, 0, len(array)-1)
	want := []int{17, 20, 26, 31, 44, 54, 55, 77, 93}
	if !reflect.DeepEqual(array, want) {
		t.Errorf("got:%v, expected:%v", array, want)
	} else {
		fmt.Println("test pass")
	}
}

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
	want := 10
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
