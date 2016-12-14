local matio = require 'matio'
-- load a single array from file
tensor = matio.load('mymat.mat', 'sample')
print (tensor)