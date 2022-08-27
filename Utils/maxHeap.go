package Utils

type MaxHeap struct {
	Size, Capacity int
	Elements       []int
}

func NewMaxHeap(capacity int) *MaxHeap {
	return &MaxHeap{
		Capacity: capacity,
		Size:     0,
		Elements: make([]int, capacity),
	}
}

func (mh *MaxHeap) Length() int {
	return mh.Size
}

// Add 向堆中添加元素
func (mh *MaxHeap) Add(value int) {
	if mh.Size >= mh.Capacity {
		return
	}
	mh.Elements[mh.Size] = value
	// 维持堆的特性
	mh.ShiftUp(mh.Size)
	mh.Size++
}

/*
比较新添加的节点与其父亲节点的值,若父亲节点的值更小,则交换位置.
*/

func (mh *MaxHeap) ShiftUp(ndx int) {
	if ndx > 0 {
		parent := (ndx - 1) / 2
		// 比较新添加的节点与其父亲节点的值,若父亲节点的值更小,则交换位置
		if mh.Elements[ndx] > mh.Elements[parent] {
			mh.Elements[ndx], mh.Elements[parent] = mh.Elements[parent], mh.Elements[ndx]
			// 递归直到父节点的值大于新添加节点值为止。
			mh.ShiftUp(parent)
		}
	}
}

// Extract 弹出并删除最大根节点
func (mh *MaxHeap) Extract() int {
	if mh.Capacity <= 0 {
		return -1
	}
	// 保存根节点值
	value := mh.Elements[0]
	mh.Size--
	// 最右下的叶子节点放到root位置
	mh.Elements[0] = mh.Elements[mh.Size]
	// 维持堆的特性
	mh.ShiftDown(0)
	return value
}

/*
将根节点与其左右节点相互比较，将最大的节点与根节点位置互换。
判断是否有左节点并且左大于根，左大于右；判断判断是否有右节点并且右大于根，右大于左
*/

func (mh *MaxHeap) ShiftDown(ndx int) {
	largest := ndx
	left := 2*ndx + 1
	right := 2*ndx + 2
	if left < mh.Size && mh.Elements[left] >= mh.Elements[largest] && mh.Elements[left] >= mh.Elements[right] {
		largest = left
	}
	if right < mh.Size && mh.Elements[right] >= mh.Elements[largest] && mh.Elements[right] >= mh.Elements[left] {
		largest = right
	}
	if largest != ndx {
		mh.Elements[ndx], mh.Elements[largest] = mh.Elements[largest], mh.Elements[ndx]
		// 递归直到根节点值为最大值为止。
		mh.ShiftDown(largest)
	}
}
