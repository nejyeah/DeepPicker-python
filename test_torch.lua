require 'torch'
require 'image'
image_2d = torch.Tensor(1950, 1950)
s = image_2d:storage()
for i=1, s:size() do
    s[i] = torch.uniform()
end

timer = torch.Timer()
step_size = 4
particle_size = 60
for i=1, 1950-particle_size, step_size do
for j=1, 1950-particle_size, step_size do
    local particle = image_2d:narrow(1,i,particle_size):narrow(2,j,particle_size):clone()
    local max = particle:max()
    local min = particle:min()
    particle:add(-min):div(max-min) 
    particle = image.scale(particle, 60, 60)
    local mean = particle:mean()
    local std = particle:std()
    particle:add(-mean)
    particle:div(std)
end
end
print("time cost:",timer:time().real)
