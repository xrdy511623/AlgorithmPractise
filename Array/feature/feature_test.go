package feature

import (
	"fmt"
	"strconv"
	"strings"
	"testing"
)

func TestMinSubArrayLen(t *testing.T) {
	nums := []int{5, 1, 3, 5, 10, 7, 4, 9, 2, 8}
	got := MinSubArrayLen(15, nums)
	want := 2
	if got != want {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}

func TestSimpleMemoryPool(t *testing.T) {
	var N int
	fmt.Scan(&N)             // 读取操作数
	mp := NewMemoryPool(100) // 初始化 100 字节内存池

	for i := 0; i < N; i++ {
		var command string
		fmt.Scan(&command)                   // 读取每条命令
		parts := strings.Split(command, "=") // 分割操作和参数
		if len(parts) != 2 {
			fmt.Println("error")
			continue
		}
		op, paramStr := parts[0], parts[1]
		switch op {
		case "REQUEST":
			size, err := strconv.Atoi(paramStr)
			if err != nil {
				fmt.Println("error")
				continue
			}
			addr, errStr := mp.Request(size)
			if errStr != "" {
				fmt.Println(errStr)
			} else {
				fmt.Println(addr)
			}
		case "RELEASE":
			addr, err := strconv.Atoi(paramStr)
			if err != nil {
				fmt.Println("error")
				continue
			}
			errStr := mp.Release(addr)
			if errStr != "" {
				fmt.Println(errStr)
			}
		default:
			fmt.Println("error")
		}
	}
}
