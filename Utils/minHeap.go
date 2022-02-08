package Utils

import (
	"errors"
)

type MinHeap struct {
	Size, Capacity int
	Elements       []int
}

func NewMinHeap(capacity int) *MinHeap {
	return &MinHeap{
		Capacity: capacity,
		Size:     0,
		Elements: make([]int, capacity),
	}
}

func (mh *MinHeap) Length() int {
	return mh.Size
}

// Add 向堆中添加元素
func (mh *MinHeap) Add(value int) {
	if mh.Size >= mh.Capacity {
		return
	}
	mh.Elements[mh.Size] = value
	// 维持堆的特性
	mh.ShiftUp(mh.Size)
	mh.Size++
}

/*
比较新添加的节点与其父亲节点的值,若父亲节点的值更大,则交换位置.
*/

func (mh *MinHeap) ShiftUp(ndx int) {
	if ndx > 0 {
		parent := (ndx - 1) / 2
		// 新添加的节点值必须比其父节点的值更大
		if mh.Elements[ndx] < mh.Elements[parent] {
			mh.Elements[ndx], mh.Elements[parent] = mh.Elements[parent], mh.Elements[ndx]
			// 递归直到父节点的值小于新添加节点值为止。
			mh.ShiftUp(parent)
		}
	}
}

// Extract 弹出并删除最小根节点
func (mh *MinHeap) Extract() (int, error) {
	if mh.Capacity <= 0 {
		err := errors.New("empty heap")
		return -1, err
	}
	// 保存根节点值
	value := mh.Elements[0]
	mh.Size--
	// 最右下的叶子节点放到root位置
	mh.Elements[0] = mh.Elements[mh.Size]
	// 维持堆的特性
	mh.ShiftDown(0)
	return value, nil
}

// ShiftDown 根节点的值必须比它的左右子节点的值都要小
func (mh *MinHeap) ShiftDown(ndx int) {
	smallest := ndx
	left := 2*ndx + 1
	right := 2*ndx + 2
	if left < mh.Size && mh.Elements[left] <= mh.Elements[smallest] && mh.Elements[left] <= mh.Elements[right] {
		smallest = left
	}
	if right < mh.Size && mh.Elements[right] <= mh.Elements[smallest] && mh.Elements[right] <= mh.Elements[left] {
		smallest = right
	}
	if smallest != ndx {
		mh.Elements[ndx], mh.Elements[smallest] = mh.Elements[smallest], mh.Elements[ndx]
		// 递归直到根节点值为最小值为止。
		mh.ShiftDown(smallest)
	}
}
