clear all;

import matlab.unittest.TestSuite;

p = pwd;

% Add the files under test to our path
addpath(p);

tests = TestSuite.fromFolder('test');

run(tests);

% Remove files under test from our path
rmpath(p);

clear p;