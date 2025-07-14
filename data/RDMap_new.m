%% RDMapParser 脉压数据解析程序
% =========================================================================
% 程序功能：
%   1. 读取雷达原始回波脉压数据（I/Q格式）
%   2. 执行动目标检测（MTD）处理
%   3. 对处理结果进行可视化展示（距离-多普勒图）
%   4. 航迹数据叠加显示
%
% 处理流程：
%   MTD加窗 → FFT处理 → 目标检测 → 结果可视化
%
% 输入数据要求：
%   1. 原始回波文件格式：'航迹批号_Label_目标类型标签.dat'
%   2. 航迹文件格式：'Tracks_航迹批号_目标类型标签_航迹长度.txt'
%   3. 点迹文件格式：'PointTracks_航迹批号_目标类型标签_航迹长度.txt'
%
% 输出结果：
%   1. 右侧坐标轴实时显示MTD处理结果（距离-多普勒谱）
%   2. 左侧坐标轴航迹叠加显示，当前原始回波数据隶属的航迹点位置
%
% 关键参数：
%   Fs = 20e6;          % 采样率 (20 MHz)
%   PRF = 1/PRT;        % 脉冲重复频率
%   delta_R = c/2/Fs;   % 距离分辨率
%   delta_V = c/2/PRF;  % 速度分辨率
%
% 使用说明：
%   1. 在程序运行后选择数据根目录，然后选择原始回波数据文件
%   2. 使用界面控件控制处理流程（暂停/继续/停止）
%
% 版权声明：
%   本程序属中国航天科工二院二十三所航天南湖公司所有，未经授权不得用于其他用途。
%   Copyright © Aerospace Nanhu Electronic Information Technology Co., Ltd.
%
% 作者：中国航天科工二院二十三所 航天南湖公司 低空事业部
% 最后修改日期：2025-06-17
% =========================================================================

clc; clear; close all;

%% 初始化状态变量
stop_flag = -1;
Fs = 20e6;              % 采样率 (20 MHz)
delta_R = 3e8/2/Fs;     % 距离分辨率

%% 读取数据

rootDir = uigetdir(pwd, '选择数据根目录（包含原始回波、点迹、航迹文件夹）');
batchFiles = getBatchFileList(rootDir);  % 结构体数组，每个元素描述一个批次
numBatches = length(batchFiles);
disp(['找到 ', num2str(numBatches), ' 个待处理批次。']);

% 创建全局输出目录（存储所有批次结果）
outputDir = fullfile(rootDir, 'MTD_Results');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%[ColPoint,ColTrack] = funcColIndex();
%[fid_rawData,pointData,trackData] = funcReadData(ColPoint,ColTrack,rootDir);

%% 创建图窗
%[mainFig,trackPlot,pointPlot,mtdImage,targetPlot] = funcCreateFigure(trackData);

%%遍历每个批次并处理
for batchIdx = 2:numBatches
    currentBatch = batchFiles(batchIdx);
    disp(['正在处理第 ', num2str(batchIdx), '/', num2str(numBatches), ' 个批次：批号 ', ...
        num2str(currentBatch.BatchNum), '，标签 ', num2str(currentBatch.Label)]);

    labelDir = fullfile(outputDir, sprintf('Label_%d', currentBatch.Label));
    if ~exist(labelDir, 'dir')
        mkdir(labelDir);
    end

    batchDir = fullfile(labelDir, sprintf('Batch_%d', currentBatch.BatchNum));
    if ~exist(batchDir, 'dir')
        mkdir(batchDir);
    end

    %% 步骤1：加载当前批次的全量数据（原始回波、点迹、航迹）
    % 读取点迹和航迹数据（全量，非逐帧）
    [ColPoint, ColTrack] = funcColIndex();
    pointData = readtable(currentBatch.PointFile, "ReadVariableNames", false);
    pointData.Properties.VariableNames = fieldnames(ColPoint);  % 列名对齐
    trackData = readtable(currentBatch.TrackFile, "ReadVariableNames", false);
    trackData.Properties.VariableNames = fieldnames(ColTrack);  % 列名对齐

    %% 步骤2：初始化当前批次的帧计数器
    frameCount = 0;  % 当前批次内的帧计数器

    %% 步骤3：逐帧处理当前批次的原始回波数据
    fid_rawData = fopen(currentBatch.RawFile, 'r');  % 打开当前批次的原始回波文件
    while ~feof(fid_rawData)
        if stop_flag == 0
            break;
        end

        % 读取单帧原始回波数据（含参数和I/Q）
        [para, data_out] = funcRawDataParser(fid_rawData);
        if isempty(para) || isempty(data_out)
            continue;  % 跳过无效帧
        end
        frameCount = frameCount + 1;

        %% 步骤4：MTD处理与目标检测
        % MTD加窗
        %MTD_win = taylorwin(size(data_out, 1), [], -30);
        %coef_MTD_2D = repmat(MTD_win, [1, size(data_out, 1)]);
        %coef_MTD_2D = permute(coef_MTD_2D, [2, 1]);
        %data_proc_MTD_win_out = data_out .* coef_MTD_2D;
        % MTD加窗（修正后）
        distance_bins = size(data_out, 1);  % 距离单元数（31）
        prt_bins = size(data_out, 2);       % PRT数（570）
        
        % 生成泰勒窗（窗长=距离单元数，1×31的行向量）
        MTD_win = taylorwin(distance_bins, [], -30);
        
        % 转换为列向量（31×1），以便在PRT维度（列方向）重复
        MTD_win_col = MTD_win(:);  % 31×1 的列向量
        
        % 在PRT维度（列方向）重复窗函数，生成与 data_out 同维度的矩阵（31×570）
        coef_MTD_2D = repmat(MTD_win_col, 1, prt_bins);  % 关键修正：列方向重复570次
        
        
        % 逐元素相乘（维度匹配）
        data_proc_MTD_win_out = data_out .* coef_MTD_2D;  % 31×570 .* 31×570


        % FFT处理生成距离-多普勒谱
        data_proc_MTD_result = fftshift(fft(data_proc_MTD_win_out, [], 2), 2);
        
        % % 杂波抑制（中心多普勒通道置零）
        % center_doppler = round(size(data_proc_MTD_result, 2)/2);
        % data_proc_MTD_result(:, center_doppler-3:center_doppler+3) = 0;

        % 计算多普勒速度轴
        delta_Vr = 3e8 / (2 * size(data_out, 2) * para.PRT * para.Freq);
        Vr = (-size(data_out, 2)/2 : size(data_out, 2)/2 - 1) * delta_Vr;

        % 目标检测
        Amp_max_Vr_unit = para.Track_No_info(4);
        Amp_max_Vr_unit = (Amp_max_Vr_unit > size(data_proc_MTD_result, 2)/2) .* ...
            (Amp_max_Vr_unit - size(data_proc_MTD_result, 2)/2) + ...
            (Amp_max_Vr_unit <= size(data_proc_MTD_result, 2)/2) .* ...
            (Amp_max_Vr_unit + size(data_proc_MTD_result, 2)/2);

        % 在截取的数据中，目标中心位于第16个距离单元
        center_local_bin = 16;
        local_radius = 5;

        % 计算局部检测窗口
        range_start_local = max(1, center_local_bin - local_radius);
        range_end_local = min(size(data_proc_MTD_result, 1), center_local_bin + local_radius);
        doppler_start = max(1, Amp_max_Vr_unit - local_radius);
        doppler_end = min(size(data_proc_MTD_result, 2), Amp_max_Vr_unit + local_radius);

        Target_sig = data_proc_MTD_result(range_start_local:range_end_local, doppler_start:doppler_end);

        % 检测峰值
        [Amp_max_index_row, Amp_max_index_col] = find(abs(Target_sig) == max(max(abs(Target_sig))), 1);

        % 获取目标全局距离单元索引
        global_range_bin = para.Track_No_info(3);

        % 计算实际距离范围（目标距离单元±15）
        range_start_bin = global_range_bin - 15; % 截取起始距离单元
        range_end_bin = global_range_bin + 15;   % 截取结束距离单元

        % 计算真实距离轴
        Range_plot = (range_start_bin:range_end_bin) * delta_R;

        % 转换到全局距离位置
        detected_range_bin = range_start_local + Amp_max_index_row - 1;
        Amp_max_range = Range_plot(detected_range_bin);
        Amp_max_Vr = Vr(doppler_start + Amp_max_index_col - 1);
        
        % 确定航迹点序号
        index_trackPointNo = min(para.Track_No_info(2), height(trackData));

        %% 步骤4：存储当前帧结果
        % 定义当前帧的输出路径（按批次+帧号分层存储）
        frameOutputDir = fullfile(batchDir, sprintf('Frame_%d', frameCount));
        if ~exist(frameOutputDir, 'dir')
            mkdir(frameOutputDir);
        end
        
        % 保存MTD处理结果（与原始算法输出一致）
        rd_matrix = data_proc_MTD_result;
        range_axis = Range_plot;
        velocity_axis = Vr;
        mat_filepath = fullfile(frameOutputDir, "MTD_result.mat");
        save(mat_filepath, 'rd_matrix', 'range_axis', 'velocity_axis', 'para', '-v7.3');
        
        % 保存目标检测结果（与原始算法检测到的目标一致）
        range = trackData.R(index_trackPointNo);
        velocity = pointData.Doppler(index_trackPointNo);
        save(fullfile(frameOutputDir, 'Target_Detection.mat'), 'range', 'velocity', '-v7.3');
    end
    
    %% 步骤5：存储当前批次的全量结果
    save(fullfile(batchDir, 'Full_Track_Data.mat'), 'trackData');
    
    %% 步骤6：关闭当前批次的资源
    fclose(fid_rawData);
end

disp('所有批次处理完成！');

%% % FUNCTIONS % %%
function [ColPoint,ColTrack] = funcColIndex()
% funcColIndex 数据列索引

% 点迹数据列索引
% 点时间、批号、距离、方位、俯仰、多普勒速度、和幅度、信噪比、原始点数量
ColPoint.Time = 1;      % 点时间
ColPoint.TrackID = 2;   % 航迹批号
ColPoint.R = 3;         % 距离
ColPoint.AZ = 4;        % 方位
ColPoint.EL = 5;        % 俯仰
ColPoint.Doppler = 6;   % 多普勒速度
ColPoint.Amp = 7;       % 和幅度
ColPoint.SNR = 8;       % 信噪比
ColPoint.PointNum = 9;  % 原始点数量

% 航迹数据列索引
% 点时间、批号、滤波距离、滤波方位、滤波俯仰、全速度、X向速度、Y向速度、Z向速度、航向
ColTrack.Time = 1;      % 点时间
ColTrack.TrackID = 2;   % 航迹批号
ColTrack.R = 3;         % 滤波距离
ColTrack.AZ = 4;        % 滤波方位
ColTrack.EL = 5;        % 滤波俯仰
ColTrack.Speed = 6;     % 全速度
ColTrack.Vx = 7;        % X向速度(东)
ColTrack.Vy = 8;        % Y向速度(北)
ColTrack.Vz = 9;        % Z向速度(天)
ColTrack.Head = 10;     % 航向角

% End of function_funcColIndex.
end

function [fid_rawData,pointData,trackData] = funcReadData(ColPoint,ColTrack,rootDir)
% funcReadData 读取数据

% 设置文件路径
if rootDir ~= 0
    IQDataDir = fullfile(rootDir, '原始回波');
    TrackDir = fullfile(rootDir, '航迹');
    PointDir = fullfile(rootDir, '点迹');

    if ~all([exist(IQDataDir,"dir"),exist(TrackDir,"dir"),exist(PointDir,"dir")])
        error('错误！未选择正确路径，根目录下需包含原始回波、点迹、航迹文件夹。');
    end
else
    error('错误！未选择文件路径。');
end

% 原始回波文件
[fileName,filePath] = uigetfile('*.dat','选择数据',IQDataDir);
rawDataFile = fullfile([filePath,fileName]);
fid_rawData = fopen(rawDataFile,'r');

% 数据批号与标签值
tokens = regexp(fileName, '^(\d+)_Label_(\d+)\.dat$', 'tokens');
tokens = tokens{1};
track_No = str2double(tokens{1});
label = str2double(tokens{2});

% 点迹文件
pointPattern = sprintf('PointTracks_%d_%d_*.txt', track_No, label);
pointFile = dir(fullfile(PointDir, pointPattern));
pointFile = fullfile(pointFile.folder,pointFile.name);
if ~exist(pointFile, 'file')
    error('点迹文件不存在: %s', pointFile);
end
pointData = readtable(pointFile,"ReadVariableNames",false);
%pointData.Properties.VariableNames(struct2array(ColPoint)) = fieldnames(ColPoint);
pointData.Properties.VariableNames = fieldnames(ColPoint);  % 直接使用结构体字段名作为列名

% 航迹文件
trackPattern = sprintf('Tracks_%d_%d_*.txt', track_No, label);
trackFile = dir(fullfile(TrackDir, trackPattern));
trackFile = fullfile(trackFile.folder,trackFile.name);
if ~exist(trackFile, 'file')
    error('航迹文件不存在: %s', trackFile);
end
trackData = readtable(trackFile,"ReadVariableNames",false);
%trackData.Properties.VariableNames(struct2array(ColTrack)) = fieldnames(ColTrack);
trackData.Properties.VariableNames = fieldnames(ColTrack);  % 直接使用结构体字段名作为列名

disp("已读取输入数据。");

% End of function_funcReadData.
end

function [mainFig,trackPlot,currentPointPlot,mtdImage,targetPlot] = funcCreateFigure(trackData)
% funcCreateFigure 创建图窗

% 创建图形界面
mainFig = figure("Name",'RDMapParser','NumberTitle','off');
set(mainFig, 'position', [100, 100, 1400, 700]);
movegui(mainFig, 'center');

uicontrol('String', '暂停', 'Position', [2, 5, 100, 30], 'Callback', 'uiwait');
uicontrol('String', '继续', 'Position', [110, 5, 100, 30], 'Callback', 'uiresume');
uicontrol('String', '停止', 'Position', [218, 5, 100, 30], 'Callback', @buttonStop);

% 创建航迹显示区域
trackAx = subplot(1, 2, 1);
trackPlot = plot(trackAx, trackData.AZ, trackData.R, '*-');
grid(trackAx, "on");
xlabel(trackAx, "方位(度)");
ylabel(trackAx, "距离(米)");
title(trackAx, "目标航迹");
hold(trackAx, "on");
currentPointPlot = plot(trackAx, NaN, NaN, 'ro'); % 用于更新当前点

% 创建MTD结果显示区域
RDMapAx = subplot(1, 2, 2);
mtdImage = imagesc(RDMapAx, [], [], []);
colorbar(RDMapAx);
xlabel(RDMapAx, "多普勒速度(米/秒)");
ylabel(RDMapAx, "距离(米)");
title(RDMapAx, "MTD处理结果");
hold(RDMapAx,"on");
targetPlot = plot(RDMapAx, NaN, NaN, 'ro', 'MarkerSize', 8);
hold(RDMapAx, "off");
set(RDMapAx, 'YDir', 'normal');

% End of function_funcCreateFigure.
end

function [para,data_out] = funcRawDataParser(fid)
% funcRawDataParser 读取解析原始回波数据

para = [];
data_out = [];

frame_head = hex2dec('FA55FA55');
frame_end = hex2dec('55FA55FA');

head_find = fread(fid,1,'uint32');
if isempty(head_find)
    return
end
while head_find ~= frame_head && ~feof(fid)
    fseek(fid,-3,'cof');
    head_find = fread(fid,1,'uint32');
    if feof(fid)
        return;
    end
end
% fseek(fid,-4,'cof');%回到起点

frame_data_length = fread(fid,1,'uint32');      %帧长度
frame_data_length = frame_data_length * 4;
fseek(fid,frame_data_length - 12,'cof');        %偏移到结尾
end_find = fread(fid,1,'uint32');               %读取帧尾
while (head_find ~= frame_head) || (end_find ~= frame_end)
    fseek(fid,-frame_data_length + 1,'cof');      % 指针偏移
    % if ftell(fid) + frame_data_length > file_byte
    %     fseek(fid,file_byte,'bof');
    %     fread(fid,1,'uint8');
    %     return
    % end
    head_find = fread(fid,1,'uint32');

    frame_data_length = fread(fid,1,'uint32');  % 通用报文头
    frame_data_length = frame_data_length * 4;

    fseek(fid,frame_data_length - 8,'cof');
    end_find = fread(fid,1,'uint32');

    if feof(fid) && (head_find ~= frame_head || end_find ~= frame_end)
        disp('未找到满足报文格式的数据');
        return;
    end
end
fseek(fid,- frame_data_length + 4,'cof');

data_temp1 = fread(fid,3,'uint32');
para.E_scan_Az = data_temp1(2) .* 0.01;
pointNum_in_bowei = data_temp1(3);

data_temp = fread(fid,pointNum_in_bowei * 4 + 5,'uint32');
para.Track_No_info = data_temp(1:pointNum_in_bowei * 4);        % [批号,航迹点序号,幅度最大值距离单元,幅度最大值多普勒单元]

% para.pointNum = data_temp(pointNum_in_bowei*2+1);             % 航迹点序号
para.Freq = data_temp(pointNum_in_bowei * 4 + 1) .* 1e6;        % 频率，单位MHz
para.CPIcount = data_temp(pointNum_in_bowei * 4 + 2);           % CPI流水号
para.PRTnum = data_temp(pointNum_in_bowei * 4 + 3);             % 当前CPI内PRT数目
para.PRT = data_temp(pointNum_in_bowei * 4 + 4) .* 0.0125e-6;   % 当前CPI内PRT数目
para.data_length = data_temp(pointNum_in_bowei*4+5);            % 距离维采样点数，采样率8MHz

data_out_temp = fread(fid,para.PRTnum * 31 * 2,'float');
if feof(fid)
    return
end
data_out_real = data_out_temp(1:2:end);
data_out_imag = data_out_temp(2:2:end);
data_out_complex = data_out_real + 1i .* data_out_imag;
data_out = reshape(data_out_complex,31,para.PRTnum);  %距离×PRT数

fseek(fid,4,'cof');

return

% End of function_funcRawDataParser.
end

function [batchFiles] = getBatchFileList(rootDir)
% getBatchFileList 自动获取批量处理文件列表（按批号和标签匹配）
% 输入：rootDir - 数据根目录（必须包含原始回波、点迹、航迹三个子文件夹）
% 输出：batchFiles - 结构体数组，每个元素描述一个待处理数据集，字段包括：
%   - BatchNum: 航迹批号（如123）
%   - Label: 目标类型标签（如0）
%   - RawFile: 原始回波文件路径（如./数据根目录/原始回波/123_Label_0.dat）
%   - PointFile: 点迹文件路径（如./数据根目录/点迹/PointTracks_123_0_100.txt）
%   - TrackFile: 航迹文件路径（如./数据根目录/航迹/Tracks_123_0_100.txt）
if  nargin < 1 || isempty(rootDir)
    rootDir = uigetdir(pwd, '选择数据根目录（需包含原始回波、点迹、航迹文件夹）');
    if rootDir == 0  % 用户取消选择
        error('未选择数据根目录！');
    end
end
batchFiles = struct('BatchNum', [], 'Label', [], 'RawFile', [], 'PointFile', [], 'TrackFile', []);

% 定义子文件夹路径（原始回波、点迹、航迹）
IQDataDir = fullfile(rootDir, '原始回波');
TrackDir = fullfile(rootDir, '航迹');
PointDir = fullfile(rootDir, '点迹');

% 检查子文件夹是否存在（避免路径错误）
if ~exist(IQDataDir, 'dir') || ~exist(TrackDir, 'dir') || ~exist(PointDir, 'dir')
    error('错误！数据根目录下需包含原始回波、点迹、航迹三个子文件夹。');
end

% 遍历原始回波文件（格式：*批号_Label_标签.dat）
rawFiles = dir(fullfile(IQDataDir, '*.dat'));
for i = 1:length(rawFiles)
    fileName = rawFiles(i).name;  % 当前文件名（如"123_Label_0.dat"）

    % 正则匹配提取批号和标签（格式：航迹批号_Label_目标类型标签.dat）
    match = regexp(fileName, '^(\d+)_Label_(\d+)\.dat$', 'tokens');
    if ~isempty(match)  % 文件名符合规则
        batchNum = str2double(match{1}{1});  % 提取批号（如123）
        label = str2double(match{1}{2});     % 提取标签（如0）
        
        % 构造点迹和航迹文件的通配符（如"PointTracks_123_0_*.txt"）
        pointPattern = sprintf('PointTracks_%d_%d_*.txt', batchNum, label);
        trackPattern = sprintf('Tracks_%d_%d_*.txt', batchNum, label);

        % 检查点迹文件是否存在（至少一个匹配项）
        pointFile = dir(fullfile(PointDir, pointPattern));
        % 检查航迹文件是否存在（至少一个匹配项）
        trackFile = dir(fullfile(TrackDir, trackPattern));

        % 若点迹和航迹文件均存在，则记录当前批次信息
        if ~isempty(pointFile) && ~isempty(trackFile)
            batchFiles(end+1).BatchNum = batchNum;       % 批号
            batchFiles(end).Label = label;               % 标签
            batchFiles(end).RawFile = fullfile(IQDataDir, fileName);  % 原始回波文件路径
            batchFiles(end).PointFile = fullfile(PointDir, pointFile(1).name);  % 点迹文件路径（取第一个匹配项）
            batchFiles(end).TrackFile = fullfile(TrackDir, trackFile(1).name);  % 航迹文件路径（取第一个匹配项）
        else
            missingPoint = isempty(pointFile);
            missingTrack = isempty(trackFile);
            msg = sprintf('警告：批号 %d、标签 %d 的', batchNum, label);
            if missingPoint && missingTrack
                msg = [msg, '点迹和航迹文件均未找到，已跳过。'];
            elseif missingPoint
                msg = [msg, '点迹文件未找到，已跳过。'];
            else
                msg = [msg, '航迹文件未找到，已跳过。'];
            end
            disp(msg);  % 输出提示到命令行

        end

    end
end

% 若未找到符合条件的批次，提示错误
if isempty(batchFiles)
    error('未找到符合命名规则的批量处理文件（需为：航迹批号_Label_目标类型标签.dat）！');
end

end