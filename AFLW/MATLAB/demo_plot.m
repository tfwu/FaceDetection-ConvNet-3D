img = imread('/home/yunzhu/face/AFW/testimages/3729198156.jpg');
img = imresize(img, [600, 924]);
figure;
imshow(img);


hold on
%points = [
%[ 339.90307617  390.78442383  307.79980469  360.11169434  415.75970459 417.52761841  330.63742065  387.43762207  370.0581665   369.80776978]
%[ 165.80751038  158.64286804  197.01716614  175.40838623  187.78433228 196.03819275  185.07225037  223.01611328  265.12127686  129.16648865]];
%scatter(points(1, :), points(2, :))


% rectangle('Position',[607.29291838225879, 248.31041036962444, 676.64922028961621 - 607.29291838225879, 338.04019097853529 - 248.31041036962444],'EdgeColor','r', 'LineWidth', 3)


rec = load('../file/rpn_bbox.txt');
shape_rec = size(rec);
for i = 1:1:160
    rectangle('Position', [rec(i, 1), rec(i, 2), rec(i, 3), rec(i, 4)], 'EdgeColor','r', 'LineWidth', 1)
end

%{
ell = load('../file/final_ell_before_nms.txt');
ell(1:1, :)
p = zeros(22551, 1);
for i = 1:1:160
    major = ell(i, 1) / 2.0;
    minor = ell(i, 2) / 2.0;
    angle = ell(i, 3);    
    center_x =  ell(i, 4);
    center_y =  ell(i, 5);
    p(i) = ellipse(major, minor, angle, center_y, center_x);
end
%}




