package sort

import (
	"fmt"
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
	array := []int{54, 26, 93, 17, 77, 31, 44, 55, 20}
	QuickSort(array, 0, len(array)-1)
	want := []int{17, 20, 26, 31, 44, 54, 55, 77, 93}
	if !reflect.DeepEqual(array, want) {
		t.Errorf("got:%v, expected:%v", array, want)
	} else {
		fmt.Println("test pass")
	}
}

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
	want := 10
	if got != want {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}
