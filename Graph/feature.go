package graph

/*
拓扑排序专题
*/

/*
leetcode 207  课程表
你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。
在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，
表示如果要学习课程 ai 则必须先学习课程  bi 。

例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。



示例 1：
输入：numCourses = 2, prerequisites = [[1,0]]
输出：true
解释：总共有 2 门课程。学习课程 1 之前，你需要完成课程 0 。这是可能的。

示例 2：
输入：numCourses = 2, prerequisites = [[1,0],[0,1]]
输出：false
解释：总共有 2 门课程。学习课程 1 之前，你需要先完成课程 0 ；并且学习课程 0 之前，你还应先完成课程 1 。这是不可能的。

提示：
1 <= numCourses <= 2000
0 <= prerequisites.length <= 5000
prerequisites[i].length == 2
0 <= ai, bi < numCourses
prerequisites[i] 中的所有课程对互不相同
*/

/*
拓扑排序 是一种用于有向图的排序方法，它要求图中所有的边必须从一个节点指向另一个节点，并且每个节点必须排在它所依赖的节点后面。
在一个有向图中，如果存在环，那么拓扑排序就不可能进行。

拓扑排序的关键特征是 每个节点（课程）在其依赖的所有节点（先修课程）完成后才会被处理。它通常用于表示类似 任务调度 这样的问题，
其中某些任务必须在其他任务之前完成。

在本问题中的应用：
课程和先修课程：题目中的每一门课程和其先修课程形成了有向图中的一条边。例如，如果课程 1 依赖于课程 0，那么就有一条从 0 到 1 的有向边。
判断能否完成所有课程：你能否完成所有课程，实质上是在问是否能给这些课程安排一个有效的学习顺序，即是否可以进行拓扑排序。如果图中没有环，
就可以完成所有课程；如果图中有环，就无法完成所有课程。

重要的概念
入度（Indegree）：
在有向图中，入度 是指一个节点（课程）有多少条边指向它（即多少门课程依赖于它）。换句话说，入度表示有多少课程必须先学完这门课程。
例如：课程 0 依赖于课程 1，那么课程 1 的入度为 1，表示课程 0 是课程 1 的依赖。
在本问题中，我们可以用入度来帮助判断哪些课程可以学习。只有入度为 0 的课程，才能被学习（即没有依赖的课程）。

出度（Outdegree）：
这是指从一个节点（课程）指向其它节点的边的数量。虽然在这个问题中主要用的是入度，但出度也很重要，它表示课程依赖的其他课程。
Kahn算法（拓扑排序的一种实现）


Kahn算法 是一种基于入度的拓扑排序算法。它的核心思想是：
初始化：首先找到所有入度为 0 的节点（课程）。这些课程没有任何先修课程，可以立即学习。
逐步学习课程：
从入度为 0 的课程中选取一个，标记它为已经处理。
然后对于这个课程依赖的所有课程（即与其相连的课程），将它们的入度减 1。
如果某个课程的入度变为 0，则表示它没有依赖其他课程，可以学习，将它加入队列。
判断是否有环：
如果在处理过程中，所有课程都被学习了（即处理的课程数等于总课程数），说明图中没有环，所有课程都可以学习。
如果有课程的入度一直不为 0，表示存在环，无法进行拓扑排序，这时返回 false。

举例说明：
假设图是这样的：
0 -> 1 -> 2

这表示：
课程 0 依赖于课程 1，
课程 1 依赖于课程 2。

入度情况：
课程 0 的入度是 0（没有课程依赖它），
课程 1 的入度是 1（课程 0 依赖它），
课程 2 的入度是 1（课程 1 依赖它）。

过程：
初始化队列：只有入度为 0 的课程（即课程 0）进入队列。
处理课程 0：处理后，课程 1 的入度减 1，变为 0，加入队列。
处理课程 1：处理后，课程 2 的入度减 1，变为 0，加入队列。
处理课程 2：所有课程都处理完成。
重要：只有 入度为 0 的课程才能被处理，代表它没有依赖的课程，可以直接开始学习。

总结：
入度：表示一个课程有多少个先修课程。只有入度为 0 的课程，才能被学习。
拓扑排序：我们通过不断减少课程的入度，找出所有可以学习的课程。最终如果所有课程都能被学习，则返回 true；
如果有课程无法学习（即存在环），则返回 false。
*/

func canFinish(numCourses int, prerequisites [][]int) bool {
	// 构建图，graph[i]存储的是所有依赖于课程i的课程，即i是它们的先修课程
	graph := make([][]int, numCourses)
	// 入度数组 indegree[i]存储的是课程i所依赖的先修课程数
	inDegree := make([]int, numCourses)
	// 填充图和入度数组
	for _, preReq := range prerequisites {
		pre, course := preReq[1], preReq[0]
		graph[pre] = append(graph[pre], course)
		inDegree[course]++
	}
	// 初始化队列queue，存储入度为0的课程(没有依赖的先修课程，可以直接学习的课程)
	queue := []int{}
	for i := 0; i < numCourses; i++ {
		if inDegree[i] == 0 {
			queue = append(queue, i)
		}
	}
	// 处理拓扑排序
	preProcessCount := 0
	for len(queue) > 0 {
		course := queue[0]
		queue = queue[1:]
		// 处理当前入度为0的课程
		preProcessCount++
		// 遍历当前课程的所有邻接课程(所有依赖于它的课程)，将它们的入度减 1
		for _, neighbor := range graph[course] {
			inDegree[neighbor]--
			if inDegree[neighbor] == 0 {
				queue = append(queue, neighbor)
			}
		}
	}
	// 如果处理的课程数量等于总课程数，说明没有环，所有课程都可以完成
	return preProcessCount == numCourses
}

/*
leetcode 210 课程表II
现在你总共有 numCourses 门课需要选，记为 0 到 numCourses - 1。给你一个数组 prerequisites ，其中
prerequisites[i] = [ai, bi] ，表示在选修课程 ai 前必须先选修 bi 。

例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示：[0,1] 。
返回你为了学完所有课程所安排的学习顺序。可能会有多个正确的顺序，你只要返回 任意一种 就可以了。如果不可能完成所有课程，
返回 一个空数组 。

示例 1：
输入：numCourses = 2, prerequisites = [[1,0]]
输出：[0,1]
解释：总共有 2 门课程。要学习课程 1，你需要先完成课程 0。因此，正确的课程顺序为 [0,1] 。

示例 2：
输入：numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
输出：[0,2,1,3]
解释：总共有 4 门课程。要学习课程 3，你应该先完成课程 1 和课程 2。并且课程 1 和课程 2 都应该排在课程 0 之后。
因此，一个正确的课程顺序是 [0,1,2,3] 。另一个正确的排序是 [0,2,1,3] 。

示例 3：
输入：numCourses = 1, prerequisites = []
输出：[0]

提示：
1 <= numCourses <= 2000
0 <= prerequisites.length <= numCourses * (numCourses - 1)
prerequisites[i].length == 2
0 <= ai, bi < numCourses
ai != bi
所有[ai, bi] 互不相同
*/

/*
思路:拓扑排序
本题与leetcode 207 课程表 几乎如出一辙，处理当前入度为0的课程其实就是课程安排的学习顺序
所以代码很容易就写出来了。
*/

func findOrder(numCourses int, prerequisites [][]int) []int {
	graph := make([][]int, numCourses)
	inDegree := make([]int, numCourses)
	for _, preReq := range prerequisites {
		pre, course := preReq[1], preReq[0]
		graph[pre] = append(graph[pre], course)
		inDegree[course]++
	}
	queue := []int{}
	for i := 0; i < numCourses; i++ {
		if inDegree[i] == 0 {
			queue = append(queue, i)
		}
	}
	order := []int{}
	for len(queue) > 0 {
		course := queue[0]
		queue = queue[1:]
		order = append(order, course)
		for _, neighbor := range graph[course] {
			inDegree[neighbor]--
			if inDegree[neighbor] == 0 {
				queue = append(queue, neighbor)
			}
		}
	}
	if len(order) == numCourses {
		return order
	}
	return []int{}
}
