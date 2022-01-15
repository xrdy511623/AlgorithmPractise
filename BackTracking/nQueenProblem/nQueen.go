package nQueenProblem

import "strings"

/*
leetcode 51. N皇后
1.1 n皇后问题研究的是如何将n个皇后放置在n×n的棋盘上，并且使皇后彼此之间不能相互攻击。
给你一个整数n ，返回所有不同的n皇后问题的解决方案。
每一种解法包含一个不同的n皇后问题的棋子放置方案，该方案中'Q'和'.'分别代表了皇后和空位。

输入：n = 4
输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
解释：如图queens1.png所示，4 皇后问题存在两个不同的解法。
示例 2：

输入：n = 1
输出：[["Q"]]

注意，皇后们的约束条件是：
不能同行
不能同列
不能同斜线
每一行，每一列都会有至少一个皇后
*/

func SolveNQueens(n int) [][]string {
	var res [][]string
	// 当n=2,3时，n皇后是无解的
	if n > 1 && n <= 3 {
		return res
	}
	// n=1时，有唯一解
	if n == 1 {
		res = append(res, []string{"Q"})
		return res
	}
	board := make([][]string, n)
	for i := 0; i < n; i++ {
		board[i] = make([]string, n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			board[i][j] = "."
		}
	}
	var backTrack func(int)
	backTrack = func(row int) {
		// 递归终止条件，如果递归到最后一行，则得到一个解
		if row == n {
			temp := make([]string, n)
			for i := 0; i < n; i++ {
				temp[i] = strings.Join(board[i], "")
			}
			res = append(res, temp)
			return
		}
		for column := 0; column < n; column++ {
			// 如果不满足皇后们的约束条件，则跳过本次循环
			if !IsValid(row, column, n, board) {
				continue
			}
			// 如果满足约束条件, 则在该位置放置一个皇后
			board[row][column] = "Q"
			// 递归
			backTrack(row + 1)
			// 回溯
			board[row][column] = "."
		}
	}
	backTrack(0)
	return res
}

// IsValid 判断在board[row][column]位置放置皇后是否满足所有约束条件
func IsValid(row, column, n int, board [][]string) bool {
	// 由于单层搜索时，每一层递归都只会选同一行里的一个元素，所以行不用去重了。
	// 检查列
	for i := 0; i < row; i++ {
		if board[i][column] == "Q" {
			return false
		}
	}
	// 检查135度对角线
	for i, j := row-1, column-1; i >= 0 && j >= 0; i, j = i-1, j-1 {
		if board[i][j] == "Q" {
			return false
		}
	}
	// 检查45度对角线
	for i, j := row-1, column+1; i >= 0 && j < n; i, j = i-1, j+1 {
		if board[i][j] == "Q" {
			return false
		}
	}
	return true
}

/*
leetcode 52. N皇后 II
1.2 n皇后问题研究的是如何将n个皇后放置在n × n的棋盘上，并且使皇后彼此之间不能相互攻击。
给你一个整数n，返回n皇后问题不同的解决方案的数量。
*/

func TotalNQueens(n int) int {
	// 当n=2,3时，n皇后是无解的
	if n > 1 && n <= 3 {
		return 0
	}
	// n=1时，有唯一解
	if n == 1 {
		return 1
	}
	board := make([][]string, n)
	for i := 0; i < n; i++ {
		board[i] = make([]string, n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			board[i][j] = "."
		}
	}
	count := 0
	var backTrack func(int)
	backTrack = func(row int) {
		// 递归终止条件，如果递归到最后一行，则得到一个解
		if row == n {
			count++
			return
		}
		for column := 0; column < n; column++ {
			// 如果不满足皇后们的约束条件，则跳过本次循环
			if !IsValid(row, column, n, board) {
				continue
			}
			// 如果满足约束条件, 则在该位置放置一个皇后
			board[row][column] = "Q"
			// 递归
			backTrack(row + 1)
			// 回溯
			board[row][column] = "."
		}
	}
	backTrack(0)
	return count
}