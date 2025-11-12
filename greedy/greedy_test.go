package greedy

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

func TestReconstructQueue(t *testing.T) {
	people := [][]int{{7, 0}, {4, 4}, {7, 1}, {5, 0}, {6, 1}, {5, 2}}
	got := ReconstructQueue(people)
	want := [][]int{{5, 0}, {7, 0}, {5, 2}, {6, 1}, {4, 4}, {7, 1}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("expected:%v, got:%v", want, got)
	} else {
		fmt.Println("test pass")
	}

}

func TestCompute(t *testing.T) {
	for {
		var n int
		_, err := fmt.Scan(&n)
		if err != nil {
			break
		}
		if n == 0 {
			continue
		}
		res := compute(n)
		fmt.Println(res)
	}
}

func TestMaxTasks(t *testing.T) {
	// 读取输入
	var n int
	fmt.Scan(&n) // 任务数量

	// 读取每个任务的 [si, ei]
	tasks := make([][2]int, n)
	for i := 0; i < n; i++ {
		fmt.Scan(&tasks[i][0], &tasks[i][1])
	}
	// 计算并输出结果
	result := maxTasks(tasks)
	fmt.Println(result)
}

func generateLargeInput(n int) []int {
	level := make([]int, n)
	for i := 0; i < n; i++ {
		level[i] = 1 // Same value to ensure it does not form valid groups
	}
	return level
}

func TestTeamCoding(t *testing.T) {
	testCases := []struct {
		name     string
		level    []int
		expected int
	}{
		{
			name:     "Example Case 1",
			level:    []int{1, 2, 3, 4},
			expected: 4,
		},
		{
			name:     "Example Case 2",
			level:    []int{5, 4, 7},
			expected: 0,
		},
		{
			name:     "Minimum Input Size",
			level:    []int{1},
			expected: 0,
		},
		{
			name:     "Only Two Employees",
			level:    []int{1, 2},
			expected: 0,
		},
		{
			name:     "Strictly Increasing Sequence",
			level:    []int{1, 3, 5, 7, 9},
			expected: 10, // (1,3,5), (1,3,7), (1,3,9), (1,5,7), (1,5,9), (1,7,9), (3,5,7), (3,5,9), (3,7,9), (5,7,9)
		},
		{
			name:     "Strictly Decreasing Sequence",
			level:    []int{9, 7, 5, 3, 1},
			expected: 10, // (9,7,5), (9,7,3), (9,7,1), (9,5,3), (9,5,1), (9,3,1), (7,5,3), (7,5,1), (7,3,1), (5,3,1)
		},
		{
			name:     "Mixed Sequence",
			level:    []int{4, 1, 3, 5, 2},
			expected: 2, // (1,3,5), (4,3,2)
		},
		{
			name:     "All Equal Values",
			level:    []int{2, 2, 2, 2},
			expected: 0,
		},
		{
			name:     "Large Input Case",
			level:    generateLargeInput(6000),
			expected: 0, // Single value repeated 6000 times does not form valid groups
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := teamCoding(tc.level)
			if result != tc.expected {
				t.Errorf("Failed %s: expected %d, got %d", tc.name, tc.expected, result)
			}
		})
	}
}

// TestFindSeats 采用表格驱动方式测试 findSeats 函数
func TestFindSeats(t *testing.T) {
	testCases := []struct {
		name     string
		input    string // 输入字符串，例如 "10001"
		expected int
	}{
		{
			name:     "Empty Input",
			input:    "",
			expected: 0,
		},
		{
			name:     "Single Empty Seat",
			input:    "0",
			expected: 1,
		},
		{
			name:     "Single Occupied Seat",
			input:    "1",
			expected: 0,
		},
		{
			name:     "All Empty Seats (Odd Count)",
			input:    "00000", // 5 个空位：可以落座于位置0,2,4 共 3 人
			expected: 3,
		},
		{
			name:     "All Empty Seats (Even Count)",
			input:    "0000", // 4 个空位：可以落座于位置0和2 共 2 人
			expected: 2,
		},
		{
			name:     "Partial Occupied - Example 1",
			input:    "10001", // 结果应为 1，如题目示例
			expected: 1,
		},
		{
			name:     "Alternating Seats - Example 2",
			input:    "0101", // 已有观众的座位隔开，无法再落座，结果 0
			expected: 0,
		},
		{
			name:     "All Occupied",
			input:    "11111",
			expected: 0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// 将输入字符串转换为整数数组
			seats := []int{}
			// 如果输入为空，直接构造空数组
			if tc.input != "" {
				parts := strings.Split(tc.input, "")
				for _, s := range parts {
					num, err := strconv.Atoi(s)
					if err != nil {
						t.Fatalf("Conversion error in test case %s: %v", tc.name, err)
					}
					seats = append(seats, num)
				}
			}
			// 为防止 findSeats 改变原始数据，每次传入副本
			result := findSeats(seats)
			if result != tc.expected {
				t.Errorf("Test case %s failed: expected %d, got %d", tc.name, tc.expected, result)
			}
		})
	}
}
