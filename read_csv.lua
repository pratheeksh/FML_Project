-- Read CSV file
require('torch')
local matio = require 'matio'
-- load a single array from file
--'cla', 'gac', 'org', 'sax', 'vio', 'cel', 'flu', 'gel', 'pia', 'tru', 'voi'
-- Split string
function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end
function tl(T)
    local count = 0
    for _ in pairs(T) do count = count + 1 end
    return count
end

local filePath = 'testcsv.csv'

-- Count number of rows and columns in file
local i = 0
for line in io.lines(filePath) do
  if i == 0 then
    COLS = #line:split(',')
  end
  i = i + 1
end

local ROWS = i - 1  -- Minus 1 because of header

-- Read data from CSV to tensor
local csvFile = io.open(filePath, 'r')
local header = csvFile:read()

--local data = torch.Tensor(ROWS, COLS)
local data = {}
local i = 0
local mattouse = cla
for line in csvFile:lines('*l') do
  i = i + 1
  local l = line:split(',')
value = {}
	if tl(l) > 2 then
		key  = l[1]..','..l[2]
		label = l[3]
	else
		key = l[1]
		label = l[2]
	end
	value[1] = key
	value[2] =  label
	data[i] = value
	print (data[i][1], data[i][2])
--	print(data[i][1])
--	print(data[i][2])
--	print(data[i][3])
  end

csvFile:close()

-- Serialize tensor
local outputFilePath = 'test.t7'
torch.save(outputFilePath, data)

-- Deserialize tensor object
local restored_data = torch.load(outputFilePath)

-- Make test
--print(data:size())
--print(restored_data:size())
