
%% ========================================================================
%% Generate channel realizations
%% ========================================================================
function [D_all,T_all,R_all,E_all,Eest_all] = gen_channel_HRRIS_SecureMEC(Nchannel,Nt,Nr,N,Ne,alpha_d,alpha_t,...
    alpha_r,alpha_e,P_ref_dB,xH_vec,xM,yM,xE,yE,channelUncertainty)


% Nchannel = 100;                             % number of channels for simulations
% Nt = 1;                                     % number of tx antennas
% Nr = 5;                                     % number of rx antennas
% N = 20;                                     % number of elements at RIS
% Ne = 2;                                     % number of EVE's antennas
% 
% alpha_d = 3.5;                              % path-loss exponent of MS-BS channel
% alpha_t = 2.2;                              % path-loss exponent of RIS-BS channel
% alpha_r = 2.8;                              % path-loss exponent of MS-RS channel
% alpha_e = 3;                                % path-loss exponent of MS-EVE channel
%     
% P_ref_dB = -30;

beta_0 = db2pow(P_ref_dB);                       % reference path-loss at 1m

Gt = db2pow(0);                             % antenna gains at BS (=1)
GRIS = db2pow(0);                           % antenna gains at RIS (=1)
GEVE = db2pow(0);                           % antenna gains at EVE (=1)

pathloss_d = @(x) beta_0*x^(-alpha_d);      % path loss of MS-BS channel
pathloss_t = @(x) beta_0*x^(-alpha_t);      % path loss of MS-RIS channel
pathloss_r = @(x) beta_0*x^(-alpha_r);      % path loss of RIS-BS channel
pathloss_e = @(x) beta_0*x^(-alpha_e);      % path loss of MS-EVE channel

%xH_vec = 10:5:100;
% xH_vec = 50;   

D_all = zeros(Nr,Nt,Nchannel);
T_all = zeros(N,Nt,Nchannel);
R_all = zeros(Nr,N,Nchannel);
E_all = zeros(Ne,Nt,Nchannel);
Eest_all = zeros(Ne,Nt,Nchannel);
for ii = 1:length(xH_vec)
    xH = xH_vec(ii);                        % RIS position on x-axis
%     xM = 45;                                % MS position on x-axis
%     yM = 2;                                 % MS position on y-axis
%     xE = 25;                                % EVE position on x-axis
%     yE = -10;                               % EVE position on y-axis
    
    dt = xH;                                % BS-RIS distance
    dr = sqrt((xH - xM)^2 + yM^2);          % RIS-MS distance
    d0 = sqrt(xM^2 + yM^2);                 % BS-MS distance
    de = sqrt((xM-xE)^2 + (yM - yE)^2);          % MS-EVE distance
    
    beta_d = pathloss_d(d0)*Gt*GRIS;        % large-scale coeff. of MS-BS channel
    beta_t = pathloss_t(dt)*Gt*GRIS;        % large-scale coeff. of MS-RIS channel
    beta_r = pathloss_r(dr)*GRIS;           % large-scale coeff. of RIS-BS channel
    beta_e = pathloss_e(de)*GEVE;           % large-scale coeff. of MS-EVE channel
    
%     randE_AoA = rand;
%     randE_AoD = rand;
    for n = 1:Nchannel
        
        % MS-BS channel
        kd = 0;
        D_NLOS = 1/sqrt(2)*(randn(Nr,Nt) + 1i*randn(Nr,Nt));    % Rayleigh fading for NLOS
        D = sqrt(beta_d) * sqrt(1/(kd+1))*D_NLOS;               % Rayleigh fading for direct BS-MS channel
        D_all(:,:,n) = D;
        
        % MS-RIS channel
        kt = 1;
        T_NLOS = 1/sqrt(2)*(randn(N,Nt) + 1i*randn(N,Nt));      % Rayleigh fading for NLOS
        T_LOS = LoS_channel(N, Nt, Nr, 1);
        T = sqrt(beta_t) * (sqrt(kt/(kt+1))*T_LOS + sqrt(1/(kt+1))*T_NLOS);
        T_all(:,:,n) = T; 
        
        % RIS-BS channel
        kr = 100;
        R_NLOS = 1/sqrt(2)*(randn(Nr,N) + 1i*randn(Nr,N));      % Rayleigh fading for NLOS
        R_LOS = LoS_channel(N, Nt, Nr, 2);
        R = sqrt(beta_r) * (sqrt(kr/(kr+1))*R_LOS + sqrt(1/(kr+1))*R_NLOS); 
        R_all(:,:,n) = R; 
    end
    for n = 1:2*Nchannel
        % MS-EVE channel
        ke = 0;
        E_NLOS = 1/sqrt(2)*(randn(Ne,Nt) + 1i*randn(Ne,Nt));    % Rayleigh fading for NLOS     
        
        AoD = 2*pi * rand;
        AoA = 2*pi * rand;
        a_tx = exp(1i * [0:Nt-1] .* pi * sin(AoD)).';
        a_rx = exp(1i * [0:Ne-1] .* pi * sin(AoA)).';
        E_LOS = a_rx*a_tx';
        
        E = sqrt(beta_e) * (sqrt(ke/(ke+1))*E_LOS + sqrt(1/(ke+1))*E_NLOS);            
        E_all(:,:,n) = E;
        
        %
        Eest = E + channelUncertainty*norm(E,'fro')*sqrt(0.5)*(randn(Ne,Nt) + 1i*randn(Ne,Nt));
        while norm(E - Eest,'fro')/norm(Eest,'fro') > channelUncertainty
            Eest = E ...
            + channelUncertainty*norm(E)*sqrt(0.5)*(randn(Ne,Nt) + 1i*randn(Ne,Nt));
        end
        Eest_all(:,:,n) = Eest;
    end
    
    % save data for simulation
%     file_name = strcat('Channel_',num2str(Nt),'x',num2str(Nr),'x',num2str(N),...
%         'x',num2str(Ne),'x',num2str(xM),'x',num2str(xH),'x',num2str(xE),'.mat');
%     save(file_name,'D_all','R_all','T_all','E_all')
end
