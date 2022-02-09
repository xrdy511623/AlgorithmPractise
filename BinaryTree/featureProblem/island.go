package featureProblem

import "AlgorithmPractise/Utils"

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
		if !InArea(grid, r, c) {
			return 1
		}
		if grid[r][c] == 0 {
			return 1
		}
		if grid[r][c] == 2 {
			return 0
		}
		grid[r][c] = 2
		return dfs(r, c-1) + dfs(r, c+1) + dfs(r-1, c) + dfs(r+1, c)
	}
	for r := 0; r < rows; r++ {
		for c := 0; c < columns; c++ {
			if grid[r][c] == 1 {
				return dfs(r, c)
			}
		}
	}
	return 0
}

func InArea(grid [][]int, r, c int) bool {
	rows, columns := len(grid), len(grid[0])
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
	var dfs func(r, c int)
	dfs = func(r, c int) {
		if r >= rows || c >= columns || r < 0 || c < 0 {
			return
		}
		if grid[r][c] != '1' {
			return
		}
		grid[r][c] = '2'
		dfs(r, c-1)
		dfs(r, c+1)
		dfs(r-1, c)
		dfs(r+1, c)
	}
	for r := 0; r < rows; r++ {
		for c := 0; c < columns; c++ {
			if grid[r][c] == 1 {
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
		if !InArea(grid, r, c) {
			return 0
		}
		if grid[r][c] != 1 {
			return 0
		}
		grid[r][c] = 2
		return 1 + dfs(r, c-1) + dfs(r, c+1) + dfs(r-1, c) + dfs(r+1, c)
	}
	maxArea := 0
	for r := 0; r < rows; r++ {
		for c := 0; c < columns; c++ {
			if grid[r][c] == 1 {
				area := dfs(r, c)
				maxArea = Utils.Max(maxArea, area)
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

func largestIsland(grid [][]int) int {
	rows, columns := len(grid), len(grid[0])
	maxLand, index := 0, 2
	record := make(map[int]int)
	var dfs func(int, int, int) int
	dfs = func(r, c, index int) int {
		if !InArea(grid, r, c) {
			return 0
		}
		if grid[r][c] != 1 {
			return 0
		}
		grid[r][c] = index
		return 1 + dfs(r, c-1, index) + dfs(r, c+1, index) + dfs(r-1, c, index) + dfs(r+1, c, index)
	}
	for r := 0; r < rows; r++ {
		for c := 0; c < columns; c++ {
			if grid[r][c] == 1 {
				area := dfs(r, c, index)
				maxLand = Utils.Max(maxLand, area)
				record[index] = area
				index++
			}
		}
	}
	maxPlus := 0
	for r := 0; r < rows; r++ {
		for c := 0; c < columns; c++ {
			if grid[r][c] == 0 {
				plus := getPlusArea(grid, r, c, record)
				maxPlus = Utils.Max(maxPlus, plus)
			}
		}
	}
	return Utils.Max(maxLand, maxPlus)
}

func getPlusArea(grid [][]int, r, c int, record map[int]int) int {
	size := 0
	seen := make(map[int]int)
	if InArea(grid, r, c-1) && grid[r][c-1] >= 2 {
		seen[grid[r][c-1]]++
	}
	if InArea(grid, r, c+1) && grid[r][c+1] >= 2 {
		seen[grid[r][c+1]]++
	}
	if InArea(grid, r-1, c) && grid[r-1][c] >= 2 {
		seen[grid[r-1][c]]++
	}
	if InArea(grid, r+1, c) && grid[r+1][c] >= 2 {
		seen[grid[r+1][c]]++
	}
	for index, _ := range seen {
		size += record[index]
	}
	size++
	return size
}