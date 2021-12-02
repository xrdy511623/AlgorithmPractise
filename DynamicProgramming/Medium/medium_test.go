package Medium

import (
	"fmt"
	"testing"
)

func TestBagProblemSimple(t *testing.T) {
	weight := []int{1, 3, 4}
	value := []int{15, 20, 30}
	getMaxValue := BagProblemSimple(weight, value, 4)
	expectedMaxValue := 35
	if getMaxValue != expectedMaxValue{
		t.Errorf("excepted:%v, got:%v", expectedMaxValue, getMaxValue)
	}
	fmt.Println(getMaxValue)
}