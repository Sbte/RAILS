clear all;

import matlab.unittest.TestSuite;

p = pwd;
p2 = fileparts(pwd);

% Add the files under test to our path
rmpath(genpath(p2));
addpath(p);

tests = TestSuite.fromFolder('test');

% Only non-MOC tests
tests = tests(find(~arrayfun(@(x) strncmp(x.Name, 'test_MOC', 8), tests)));
run(tests);

% Remove files under test from our path
rmpath(p);

clear p;