package permute

import (
	"sort"
	"testing"
)

func TestPermuteOfString(t *testing.T) {
	testCases := []struct {
		name     string
		input    string
		expected []string
	}{
		{
			name:     "Empty String",
			input:    "",
			expected: []string{""},
		},
		{
			name:     "Single Character",
			input:    "a",
			expected: []string{"a"},
		},
		{
			name:     "Distinct Characters",
			input:    "abc",
			expected: []string{"abc", "acb", "bac", "bca", "cab", "cba"},
		},
		{
			name:  "Duplicate Characters",
			input: "aab",
			// 排序后 "aab"，唯一排列有 "aab", "aba", "baa"
			expected: []string{"aab", "aba", "baa"},
		},
		{
			name:     "All Duplicate Characters",
			input:    "aaa",
			expected: []string{"aaa"},
		},
	}

	// 为了不依赖排列生成顺序，先排序结果再比较
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := permuteOfString(tc.input)
			sort.Strings(result)
			sort.Strings(tc.expected)
			if len(result) != len(tc.expected) {
				t.Errorf("Test %s failed: expected %v, got %v", tc.name, tc.expected, result)
			} else {
				for i := range result {
					if result[i] != tc.expected[i] {
						t.Errorf("Test %s failed: expected %v, got %v", tc.name, tc.expected, result)
						break
					}
				}
			}
		})
	}
}
