clear
clc

files = dir('*.psp8');
for id = 1:length(files)
    [~, f,ext] = fileparts(files(id).name);
    rename = strcat(f(1:2),ext) ;
    movefile(files(id).name, rename);
end