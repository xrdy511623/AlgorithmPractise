package feature

import (
	"algorithm-practise/utils"
)

/*
leetcode 463. 岛屿的周长
1.1 给定一个row x col的二维网格地图grid ，其中：grid[i][j] = 1 表示陆地， grid[i][j] = 0 表示水域。
网格中的格子水平和垂直方向相连（对角线方向不相连）。整个网格被水完全包围，但其中恰好有一个岛屿（或者说，
一个或多个表示陆地的格子相连组成的岛屿）。

岛屿中没有“湖”（“湖” 指水域在岛屿内部且不和岛屿周围的水相连）。格子是边长为1的正方形。网格为长方形，且宽度
和高度均不超过100 。计算这个岛屿的周长。

输入：grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]
输出：16

提示：
m == grid.length
n == grid[i].length
1 <= m, n <= 100
grid[i][j] 的值为 0 或 1
*/

func islandPerimeter(grid [][]int) int {
	rows, columns := len(grid), len(grid[0])
	var dfs func(int, int) int
	dfs = func(r, c int) int {
		// 越界，代表一条网格边界的边
		if !InArea(r, c, rows, columns) {
			return 1
		}
		// 遍历到海洋格子，代表一条挨着海洋的边
		if grid[r][c] == 0 {
			return 1
		}
		// 遍历到访问过的陆地格子，返回边长自然是0
		if grid[r][c] == 2 {
			return 0
		}
		// 标记已经访问过的陆地格子
		grid[r][c] = 2
		return dfs(r, c-1) + dfs(r, c+1) + dfs(r-1, c) + dfs(r+1, c)
	}
	for r := 0; r < rows; r++ {
		for c := 0; c < columns; c++ {
			// 题目限制只有一个岛屿，计算一个即可
			if grid[r][c] == 1 {
				return dfs(r, c)
			}
		}
	}
	return 0
}

// InArea 确定grid[r][c]是否越界
func InArea(r, c, rows, columns int) bool {
	return r >= 0 && r < rows && c >= 0 && c < columns
}

/*
leetcode 200. 岛屿数量
1.2 给你一个由'1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
此外，你可以假设该网格的四条边均被水包围。

示例 1：
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1

提示：
m == grid.length
n == grid[i].length
1 <= m, n <= 300
grid[i][j] 的值为 '0' 或 '1'

*/

func numIslands(grid [][]byte) int {
	rows, columns, islands := len(grid), len(grid[0]), 0
	var dfs func(int, int)
	dfs = func(r, c int) {
		// 判断base case, 此时grid[r][c]越界，无需继续遍历
		if r >= rows || c >= columns || r < 0 || c < 0 {
			return
		}
		// 如果grid[r][c]不是岛屿，直接返回
		if grid[r][c] != '1' {
			return
		}
		// 将grid[r][c]标记为已经访问过，去重
		grid[r][c] = '2'
		// 遍历上下左右四个相邻节点,若为陆地，则是同一岛屿
		dfs(r, c-1)
		dfs(r, c+1)
		dfs(r-1, c)
		dfs(r+1, c)
	}
	for r := 0; r < rows; r++ {
		for c := 0; c < columns; c++ {
			if grid[r][c] == '1' {
				islands++
				dfs(r, c)
			}
		}
	}
	return islands
}

/*
leetcode 695. 岛屿的最大面积
1.3 给你一个大小为m x n的二进制矩阵grid 。
岛屿是由一些相邻的1(代表土地) 构成的组合，这里的相邻要求两个1必须在水平或者竖直的四个方向上相邻。你可以假设
grid的四个边缘都被 0（代表水）包围着。
岛屿的面积是岛上值为1的单元格的数目。
计算并返回grid中最大的岛屿面积。如果没有岛屿，则返回面积为 0 。

示例1:
输入：grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],
[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],
[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
输出：6
解释：答案不应该是11 ，因为岛屿只能包含水平或垂直这四个方向上的 1 。

提示：
m == grid.length
n == grid[i].length
1 <= m, n <= 50
grid[i][j] 为 0 或 1
*/

func maxAreaOfIsland(grid [][]int) int {
	rows, columns := len(grid), len(grid[0])
	var dfs func(int, int) int
	dfs = func(r, c int) int {
		// 判断base case, 此时grid[r][c]越界，返回的面积自然是0
		if !InArea(r, c, rows, columns) {
			return 0
		}
		// 此时grid[r][c]不是未访问(遍历)的陆地，返回的面积自然也是0
		if grid[r][c] != 1 {
			return 0
		}
		// 将grid[r][c]标记为已访问(遍历)的陆地
		grid[r][c] = 2
		// 此时岛屿的面积就是1+相邻上下左右四个节点的面积之和
		return 1 + dfs(r, c-1) + dfs(r, c+1) + dfs(r-1, c) + dfs(r+1, c)
	}
	maxArea := 0
	for r := 0; r < rows; r++ {
		for c := 0; c < columns; c++ {
			if grid[r][c] == 1 {
				area := dfs(r, c)
				// 迭代最大岛屿面积
				maxArea = utils.Max(maxArea, area)
			}
		}
	}
	return maxArea
}

/*
leetcode 827. 最大人工岛
1.4 给你一个大小为 n x n 二进制矩阵 grid 。最多只能将一格0变成1 。
返回执行此操作后，grid中最大的岛屿面积是多少？
岛屿由一组上、下、左、右四个方向相连的1形成。

示例1:
输入: grid = [[1, 0], [0, 1]]
输出: 3
解释: 将一格0变成1，最终连通两个小岛得到面积为 3 的岛屿。

提示：
n == grid.length
n == grid[i].length
1 <= n <= 500
grid[i][j] 为 0 或 1
*/

/*
两遍DFS：第一遍DFS遍历陆地格子，计算每个岛屿的面积并标记岛屿索引index(根据索引index可以在map中找到对应的岛屿面积)；
第二遍DFS遍历海洋格子，观察每个海洋格子相邻的陆地格子，得出填海后相邻两个岛屿的面积之和。

*/

func largestIsland(grid [][]int) int {
	rows, columns := len(grid), len(grid[0])
	maxLand, index := 0, 2
	// record记录第一轮DFS遍历后的岛屿索引以及索引对应的岛屿面积
	record := make(map[int]int)
	var dfs func(int, int, int) int
	dfs = func(r, c, index int) int {
		// 判断base case, 此时grid[r][c]越界，返回的面积自然是0
		if !InArea(r, c, rows, columns) {
			return 0
		}
		// 此时grid[r][c]不是未访问(遍历)的陆地，返回的面积自然也是0
		if grid[r][c] != 1 {
			return 0
		}
		// 标记已访问(遍历)的岛屿的索引
		grid[r][c] = index
		// grid[r][c]相邻的上下左右四个节点如果也是陆地，索引自然也是一样的。
		return 1 + dfs(r, c-1, index) + dfs(r, c+1, index) + dfs(r-1, c, index) + dfs(r+1, c, index)
	}
	for r := 0; r < rows; r++ {
		for c := 0; c < columns; c++ {
			if grid[r][c] == 1 {
				area := dfs(r, c, index)
				maxLand = utils.Max(maxLand, area)
				record[index] = area
				index++
			}
		}
	}
	maxPlus := 0
	for r := 0; r < rows; r++ {
		for c := 0; c < columns; c++ {
			if grid[r][c] == 0 {
				plus := getPlusArea(grid, r, c, rows, columns, record)
				maxPlus = utils.Max(maxPlus, plus)
			}
		}
	}
	// 有可能全是陆地，所以最大岛屿面积得是maxLand, maxPlus的较大值
	return utils.Max(maxLand, maxPlus)
}

func getPlusArea(grid [][]int, r, c, rows, columns int, record map[int]int) int {
	size := 0
	seen := make(map[int]int)
	// 如果相邻节点是岛屿(第一轮DFS遍历后岛屿都标记了index,至少都是2)
	// 将标记的索引index添加到去重的哈希表seen中,因为相同索引意味着是同一个岛，不能重复累加海洋格子旁边的岛屿
	if InArea(r, c-1, rows, columns) && grid[r][c-1] >= 2 {
		seen[grid[r][c-1]]++
	}
	if InArea(r, c+1, rows, columns) && grid[r][c+1] >= 2 {
		seen[grid[r][c+1]]++
	}
	if InArea(r-1, c, rows, columns) && grid[r-1][c] >= 2 {
		seen[grid[r-1][c]]++
	}
	if InArea(r+1, c, rows, columns) && grid[r+1][c] >= 2 {
		seen[grid[r+1][c]]++
	}
	// 相邻岛屿的面积之和+填的海洋格子的面积(1)就是填海造陆后的面积
	for index, _ := range seen {
		size += record[index]
	}
	size++
	return size
}

/*
leetcode 79 单词搜索
给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的
字母不允许被重复使用。

输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true

提示：
m == board.length
n = board[i].length
1 <= m, n <= 6
1 <= word.length <= 15
board 和 word 仅由大小写英文字母组成

进阶：你可以使用搜索剪枝的技术来优化解决方案，使其在 board 更大的情况下可以更快解决问题？
*/

/*
解题思路
给定一个二维字符网格 board 和一个字符串 word，任务是判断 word 是否可以从网格中的某个位置出发，按字母顺序通过相邻
的单元格（水平或垂直）构成。每个单元格只能使用一次。

关键要点：
相邻单元格：可以通过水平或垂直相邻的单元格连接。
搜索方式：这是一个典型的“单词搜索”问题，可以用深度优先搜索（DFS）来解决。
剪枝优化：DFS 搜索过程中，如果当前字母与目标字母不匹配，或者已经走过的路径包含该位置，则可以剪枝避免无效的搜索。

解决方案
DFS 搜索：
从 board 的每一个单元格出发，进行深度优先搜索。每次搜索时，检查当前位置的字母是否与 word 中当前字母匹配。
每个单元格只能使用一次，搜索过程中要标记已经访问过的单元格，避免重复使用。
如果找到 word 中所有字母的匹配，则返回 true，否则继续搜索。

剪枝策略：
每次递归时，如果当前单元格的字母与 word 不匹配，立即返回 false。
在每次递归调用后，需要将访问状态恢复，以便其他路径能够正确使用该单元格。

边界条件：
确保在 DFS 搜索时不越界。
如果找到匹配的单词，直接返回 true。
否则继续探索。
*/

func exist(board [][]byte, word string) bool {
	// 获取网格的行列数
	m, n := len(board), len(board[0])
	// 创建一个二维数组记录网格board每个位置(单元格)的访问状态
	visited := make([][]bool, m)
	for i := 0; i < m; i++ {
		visited[i] = make([]bool, n)
	}
	// 遍历网格中的每个单元格
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			// 如果当前单元格的字母与 word[0] 相同，则从该位置开始DFS搜索
			if board[i][j] == word[0] {
				if dfs(board, word, visited, 0, i, j) {
					return true
				}
			}
		}
	}
	// 如果没有找到匹配的单词，返回false
	return false
}

func dfs(board [][]byte, word string, visited [][]bool, index, row, col int) bool {
	// 如果已经遍历完整个单词 word，返回true，表示匹配成功
	if index == len(word) {
		return true
	}
	// 边界条件，越界直接返回false
	if row < 0 || col < 0 || row >= len(board) || col >= len(board[0]) {
		return false
	}
	// 如果该位置已经访问过或者当前字符不等于目标字符，直接返回false
	if visited[row][col] || board[row][col] != word[index] {
		return false
	}
	// 标记当前位置已经访问过
	visited[row][col] = true
	// 递归，从当前位置[row][col]的相邻位置(上下左右)判断能否继续匹配
	if dfs(board, word, visited, index+1, row-1, col) {
		return true
	}
	if dfs(board, word, visited, index+1, row+1, col) {
		return true
	}
	if dfs(board, word, visited, index+1, row, col-1) {
		return true
	}
	if dfs(board, word, visited, index+1, row, col+1) {
		return true
	}
	// 回溯，在每次递归调用后，需要将访问状态恢复，以便其他路径能够正确使用该单元格。
	visited[row][col] = false
	// 如果还是没匹配成功，最后返回false
	return false
}
