package feature

import (
	"fmt"
	"testing"
)

func TestAddBase36(t *testing.T) {
	got := AddBase36("1b", "2x")
	want := "48"
	if got != want {
		t.Errorf("got:%v, expected:%v", got, want)
	} else {
		fmt.Println("test pass")
	}
}

func TestHexToDecimal(t *testing.T) {
	var hex string
	_, err := fmt.Scan(&hex)
	if err != nil {
		return
	}
	res := hexToDecimal(hex)
	fmt.Println(res)
}

func TestReplaceVariables(t *testing.T) {
	// 读取输入的 CSV 字符串
	var input string
	fmt.Scanln(&input)

	res := ReplaceVariables(input)
	// 将替换后的单元格用逗号连接并输出
	fmt.Println(res)
}
