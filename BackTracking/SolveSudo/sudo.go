package SolveSudo

/*
leetcode 37. 解数独
1.1 编写一个程序，通过填充空格来解决数独问题。
数独的解法需遵循如下规则：
数字1-9在每一行只能出现一次。
数字1-9在每一列只能出现一次。
数字1-9在每一个以粗实线分隔的3x3宫内只能出现一次。（请参考示例图）
数独部分空格内已填入了数字，空白格用'.'表示。

示例:
sudo.png

解答:
solve.png
*/

func SolveSudoku(board [][]byte) {
	var backTrack func() bool
	backTrack = func() bool {
		for i := 0; i < 9; i++ {
			for j := 0; j < 9; j++ {
				// 跳过已经填了数字的
				if board[i][j] != '.' {
					continue
				}
				for k := '1'; k <= '9'; k++ {
					if IsValidNumber(board, i, j, byte(k)) {
						// 如果验证填入k后满足约束条件，则该位置可填入k
						board[i][j] = byte(k)
						// 递归
						if backTrack() {
							return true
						}
						// 如果递归结果为false,回溯
						board[i][j] = '.'
					}
				}
				// 如果遍历完1-9都无法找到合适的数字，返回false
				return false
			}
		}
		// 如果递归完了没有返回false，说明所有空格都填上了，返回true
		return true
	}
	backTrack()
}

func IsValidNumber(board [][]byte, row, column int, value byte) bool {
	// 检查同一行有无重复
	for col := 0; col < 9; col++ {
		if board[row][col] == value {
			return false
		}
	}
	// 检查同一列有无重复
	for r := 0; r < 9; r++ {
		if board[r][column] == value {
			return false
		}
	}
	// 检查3*3=9宫格范围内有无重复
	startRow, startColumn := row/3*3, column/3*3
	for i := startRow; i < startRow+3; i++ {
		for j := startColumn; j < startColumn+3; j++ {
			if board[i][j] == value {
				return false
			}
		}
	}
	return true
}
