package feature

import (
	"bufio"
	"fmt"
	"os"
	"reflect"
	"strconv"
	"testing"
)

func TestTopKFrequent(t *testing.T) {
	nums := []int{1, 1, 1, 1, 2, 2, 3}
	get := TopKFrequent(nums, 2)
	want := []int{1, 2}
	if !reflect.DeepEqual(get, want) {
		t.Errorf("excepted:%v, got:%v", want, get)
		return
	}
	fmt.Println("pass")
}

func TestMasterWordsCount(t *testing.T) {
	// 创建一个 scanner 用于从标准输入读取数据
	scanner := bufio.NewScanner(os.Stdin)
	// 读取 words 数组的个数 N
	scanner.Scan()
	N, _ := strconv.Atoi(scanner.Text()) // 转换为整数

	// 读取 N 个单词到数组 words 中
	words := make([]string, N)
	for i := 0; i < N; i++ {
		scanner.Scan()
		words[i] = scanner.Text()
	}

	// 读取字符串 chars
	scanner.Scan()
	chars := scanner.Text()
	cnt := masterWordsCount(words, chars)
	fmt.Println(cnt)
}
