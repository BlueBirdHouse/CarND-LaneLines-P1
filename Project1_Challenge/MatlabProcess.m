function [ Lines ] = MatlabProcess( eye_Lines )
%MatlabProcess ����ִ��Ѱ������ľ�ϸ������
%   ����ִ��Ѱ������ľ�ϸ������
eye_Lines = double(eye_Lines);
Point1 = eye_Lines(:,1:2);
Point2 = eye_Lines(:,3:4);
%��������һ��б�ʹ���
Diff21 = Point2 - Point1;
ks = Diff21(:,2)./Diff21(:,1);
Angles = rad2deg(atan(ks));
%�����ǶԳƵģ��������Ǽ���
Angles = abs(Angles);
Filter = (Angles > 15)&(Angles < 75);
Point1 = Point1(Filter,:);
Point2 = Point2(Filter,:);
Point = [Point1 ; Point2];
Lines = polyLine(Point(:,1),Point(:,2));
end

function [Line] = polyLine(X_List,Y_List)
    %���ò�ֵ��ϵķ�����������Ϻ�ֱ�ߵ�����ʽ  
    p = polyfit(X_List,Y_List,1);
    Max_X = int32(max(X_List));
    Max_Y = int32(polyval(p,Max_X));
    
    Min_X = int32(min(X_List));
    Min_Y = int32(polyval(p,Min_X));
    
    [Max_X,Max_Y] = FixOutRange(Max_X,Max_Y,p);
    [Min_X,Min_Y] = FixOutRange(Min_X,Min_Y,p);
    
    Line = [Max_X Max_Y Min_X Min_Y];
    
    %�����ô���
    plot(X_List,Y_List,'o');
    hold on;
    plot([Max_X Min_X],[Max_Y Min_Y]);
end

function [X,Y] = FixOutRange(X,Y,p)
    %�趨ͼ��Ĵ�С���Է�ֹ�㷨����ͼ����ĵ�
    X_Range = 1280;
    Y_Range = 720;
    if(Y >= Y_Range)
        Y = Y_Range - 1;
        X = (Y - p(2))/p(1);
    end
    if(Y <= 0)
        Y = 1;
        X = (Y - p(2))/p(1);
    end
    X = int32(X);
    Y = int32(Y);
end
