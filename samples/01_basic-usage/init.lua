local Ort = require "luaort"

-- create a new session and load the specific model.
--
-- the model in this example contains a single MatMul node
-- it has 2 inputs: 'a'(FLOAT, 3x4) and 'b'(FLOAT, 4x3)
-- it has 1 output: 'c'(FLOAT, 3x3)
local Env = Ort.CreateEnv()
local SessionOptions = Ort.CreateSessionOptions()
local Session = Env:CreateSession("model.onnx", SessionOptions)

local dataA = ('f'):rep(12):pack(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) --pack 12 floats to bytestring
local tensorA = Ort.CreateValue({ 3, 4 }, "FLOAT", dataA)
local dataB = ('f'):rep(12):pack(10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120)
local tensorB = Ort.CreateValue({ 4, 3 }, "FLOAT", dataB)

local result = Session:Run {
	a = tensorA,
	b = tensorB
}

local dataC = result.c:GetData()

print("data of result tensor 'c'", ('f'):rep(9):unpack(dataC)) -- unpack 9 flots from bytestring