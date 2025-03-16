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
