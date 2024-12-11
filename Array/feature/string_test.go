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
