"""
Complete the solution so that it returns true if the first argument(string)
passed in ends with the 2nd argument (also a string).

Examples:

    solution('abc', 'bc') # returns true
    solution('abc', 'd') # returns false
"""

"""
Create unit test using those cases:
fixed_tests_True = (
    ( "samurai", "ai"    ),
    ( "ninja",   "ja"    ),
    ( "sensei",  "i"     ),
    ( "abc",     "abc"   ),
    ( "abcabc",  "bc"    ),
    ( "fails",   "ails"  ),
)

fixed_tests_False = (
    ( "sumo",    "omo"   ),
    ( "samurai", "ra"    ),
    ( "abc",     "abcd"  ),
    ( "ails",    "fails" ),
    ( "this",    "fails" ),
    ( "spam",    "eggs"  )
)
"""
def solution(string, ending):
    return string.endswith(ending)

import unittest

class TestSolution(unittest.TestCase):
    
    def test_fixed_tests_true(self):
        fixed_tests_True = (
            ( "samurai", "ai"    ),
            ( "ninja",   "ja"    ),
            ( "sensei",  "i"     ),
            ( "abc",     "abc"   ),
            ( "abcabc",  "bc"    ),
            ( "fails",   "ails"  ),
        )
        for string, ending in fixed_tests_True:
            with self.subTest(f"{string} ends with {ending}"):
                self.assertTrue(solution(string, ending))

    def test_fixed_tests_false(self):
        fixed_tests_False = (
            ( "sumo",    "omo"   ),
            ( "samurai", "ra"    ),
            ( "abc",     "abcd"  ),
            ( "ails",    "fails" ),
            ( "this",    "fails" ),
            ( "spam",    "eggs"  )
        )
        for string, ending in fixed_tests_False:
            with self.subTest(f"{string} does not end with {ending}"):
                self.assertFalse(solution(string, ending))


if __name__ == "__main__":
    unittest.main(exit=False)