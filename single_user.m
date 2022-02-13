% Minimization of computation offloading latency in a secure Mobile Edge 
% Computing (MEC) system assisted with Hybrid Relayi-Reflecting Intelligent 
% Surfaces (HRRIS).
% 
% Simulation code of the paper:
%
% [1] K. -H. Ngo, N. T. Nguyen, T. Q. Dinh, T. -M. Hoang and M. Juntti, 
% "Low-Latency and Secure Computation Offloading Assisted by Hybrid 
% Relay-Reflecting Intelligent Surface," 2021 International Conference on 
% Advanced Technologies for Communications (ATC), 2021, pp. 306-311.
%
% Four methods for the intelligent surface:
% 1. Fixed HRRIS
% 2. Dynamic HRRIS
% 3. RIS with random phases
% 4. RIS with optimized phases according to the paper below
% 
% [2] T. Bai, C. Pan, Y. Deng, M. Elkashlan, A. Nallanathan, and L. Hanzo,
% “Latency minimization for intelligent reflecting surface aided mobile edge
% computing,” IEEE J. Sel. Topics Signal Process., vol. 38, no. 11, pp.
% 2666–2682, Nov. 2020.
%
% Single-user scenario:
%         
%                 |           Eavesdropper (EVE)
%                 | <-----------> O
%                 |     xEVE      ^
%                 |               |    User Equipment (UE)
%                 | <-------------+-------> O
%                 |     xU        |         ^
%                 |           yEVE|       yU|
%                 |               |         |
%                 |     xH        v         v
%                 O <--------------------------------> O
%            Edge Node (EN)                          HRRIS      

clear 

%% --- Location ---
xH = 50;            % distance from EN to HRRIS
xU_set = 45; %10:5:100; % horizontal distance from UE to EN
yU = 2;             % vertical distance from UE to the EN-HRRIS line
xEVE_set = 30; %10:5:100;        % horizontal distance from EVE to EN
yEVE = 5;           % vertical distance from EVE to the EN-HRRIS line

%% --- Communication ---
M = 5;                      % #EN's antennas
E_set = 1;                  % #EVE's antennas

bandwidth = 1e6;    

d0 = 1;                     % reference distance
pathloss0 = -30;            % path loss at reference distance (dB)

% various pathloss exponents
pathlossExp_UE_AP = 3.5;    
pathlossExp_UE_EVE = 2.8; 
pathlossExp_UE_HRRIS = 2.2;
pathlossExp_HRRIS_AP = 2.2;

% pathloss function
pathloss = @(d,pathloss0,pathlossExp) 10.^(pathloss0./10).*(d./d0).^(-pathlossExp);

% uncertainty of EN about h_UE_EVE
channelUncertainty = 0.1; 

%% --- HRRIS ---
N_set = 50:10:200;                  % number of HRRIS elements
N_relay_set = 1;                    % number of active relaying elements
Upsilon = @(alpha) diag(alpha);     % HRRIS's coefficients
% Psi = @(alpha) diag(alpha(:).*Idx_relay_bin(:));
% Phi = @(alpha) diag(alpha(:).*Idx_reflect_bin(:));
    
%% --- Powers ---
PdB_set = 30; %5:5:100;         % total power in dB
P_relay_max = db2pow(0);        % power budget for the active relaying elements
noiseVar = db2pow(-80); %3.98e-12;  % noise power (mW)

%% --- Computing ---
L = 300e3; %1000*(250 + rand*(350-250));    % #bits to be processed
nu = 750; %700 + rand*(800-700);            % required #CPU cycles/bit
fl_set = 5*1e8; %(4+rand*(6-4))*1e8;        % computational capability of the user #CPU cycles/second
fe_set = 20e9; %[0 5:5:50 100:100:600]*1e8; % total edge computing capability

%% --- Monte-Carlo Simulation Parameters ---
% convergeThres = 1e-3;     % convergence criterion
nIterMC = 1000;              % number of Monte-Carlo simulations for the channels
nIterAO = 6;                % number of iterations in alternating optimization
    
%% --- Initialize the optimized latency and primary rate for difference reflecting modes ---
% RIS with random phases
latency_randomPhase = zeros(length(N_set),length(N_relay_set),length(PdB_set),...
    length(xU_set),length(xEVE_set),length(E_set),length(fe_set),length(fl_set));
primRate_randomPhase = zeros(length(N_set),length(N_relay_set),length(PdB_set),...
    length(xU_set),length(E_set),length(xEVE_set));

% RIS with optimized phases
latency_RIS = zeros(length(N_set),length(N_relay_set),length(PdB_set),...
    length(xU_set),length(xEVE_set),length(E_set),length(fe_set),length(fl_set));
primRate_RIS = zeros(length(N_set),length(N_relay_set),length(PdB_set),...
    length(xU_set),length(E_set),length(xEVE_set));

% fixed HRRIS
latency_fixed = zeros(length(N_set),length(N_relay_set),length(PdB_set),...
    length(xU_set),length(xEVE_set),length(E_set),length(fe_set),length(fl_set));
primRate_fixed = zeros(length(N_set),length(N_relay_set),length(PdB_set),...
    length(xU_set),length(E_set),length(xEVE_set));

% dynamic HRRIS
latency_dynamic = zeros(length(N_set),length(N_relay_set),length(PdB_set),...
    length(xU_set),length(xEVE_set),length(E_set),length(fe_set),length(fl_set));
primRate_dynamic = zeros(length(N_set),length(N_relay_set),length(PdB_set),...
    length(xU_set),length(E_set),length(xEVE_set));

% d_UE_AP = sqrt(d_UE^2 + dbar_UE^2);
% d_UE_HRRIS = sqrt((d_AP_HRRIS-d_UE)^2 + dbar_UE^2);
% d_UE_EVE = sqrt((d_UE-d_EVE)^2 + (dbar_UE-dbar_EVE)^2);

%% --- Simulation ---
for idxE = 1:length(E_set)
E = E_set(idxE);
for idxdUE = 1:length(xU_set)
xU = xU_set(idxdUE);
for idxdEVE = 1:length(xEVE_set)
xEVE = xEVE_set(idxdEVE);
for idxP = 1:length(PdB_set)
P = db2pow(PdB_set(idxP));
for idxN = 1:length(N_set)
N = N_set(idxN);
for idxNrelay = 1:length(N_relay_set)
    
N_relay = N_relay_set(idxNrelay);   % number of active relaying elements
N_reflect = N - N_relay;            % number of passive elements

P_UE = P - P_relay_max*(N_relay>0); % transmit power of the user (mW)

% Generate channel realizations
[channel_UE_EN,channel_UE_HRRIS,channel_HRRIS_EN,channel_UE_EVE,channel_UE_EVE_est] = ...
    gen_channel_HRRIS_SecureMEC(nIterMC,1,M,N,E,pathlossExp_UE_AP,pathlossExp_UE_HRRIS,...
        pathlossExp_HRRIS_AP,pathlossExp_UE_EVE,pathloss0,xH,xU,yU,...
        xEVE,yEVE,channelUncertainty);

% delay_set = zeros(nIterAO,1);

primRate_randomPhase_tmp = 0;
primRate_RIS_tmp = 0;
primRate_fixed_tmp = 0;
primRate_dynamic_tmp = 0;

latency_randomPhase_tmp = zeros(size(fe_set));
latency_RIS_tmp = zeros(size(fe_set));
latency_fixed_tmp = zeros(size(fe_set));
latency_dynamic_tmp = zeros(size(fe_set));

for idxMC = 1:nIterMC
    % Initialize HRRIS config
    Idx_relay = 1:N_relay;        % position of active elements
    Idx_reflect = N_relay+1:N;    % position of passive elements
    
    Idx_relay_bin = zeros(N,1);  
    Idx_reflect_bin = zeros(N,1);
    Idx_relay_bin(Idx_relay) = 1;      
    Idx_reflect_bin(Idx_reflect) = 1;   
    
    Psi = @(alpha) diag(alpha(:).*Idx_relay_bin(:));
    Phi = @(alpha) diag(alpha(:).*Idx_reflect_bin(:));
    
    %% Generate the channels
    
    % option 1: 
    h_UE_AP = channel_UE_EN(:,:,idxMC);
    h_UE_HRRIS = channel_UE_HRRIS(:,:,idxMC);
    H_EN_HRRIS = channel_HRRIS_EN(:,:,idxMC);
%     h_UE_EVE = channel_UE_EVE(:,:,idxH);
    
    % option 2:
%     d_UE_AP = sqrt(d_UE^2 + dbar_UE^2);
%     h_UE_AP = sqrt(pathloss(d_UE_AP,pathloss0,pathlossExp_UE_AP)/2)...
%                 *(randn(M,1) + 1i*randn(M,1));
%     
%     
%     d_UE_HRRIS = sqrt((d_AP_HRRIS-d_UE)^2 + dbar_UE^2);
%     h_UE_HRRIS = sqrt(pathloss(d_UE_HRRIS,pathloss0,pathlossExp_UE_HRRIS)/2)...
%                 *(randn(N,1) + 1i*randn(N,1));
% %     hh_UE_HRRIS = h_UE_HRRIS*h_UE_HRRIS';
%     
%             
%     H_AP_HRRIS = sqrt(pathloss(d_AP_HRRIS,pathloss0,pathlossExp_HRRIS_AP)/2)...
%                 *(randn(M,N) + 1i*randn(M,N));
%     
% 
%     d_UE_EVE = sqrt((d_UE-d_EVE)^2 + (dbar_UE-dbar_EVE)^2);
%     h_UE_EVE = sqrt(pathloss(d_UE_EVE,pathloss0,pathlossExp_UE_EVE)/2)...
%                 *(randn(E,1) + 1i*randn(E,1));   
    
%     hest_UE_EVE = h_UE_EVE ...
%         + channelUncertainty*norm(h_UE_EVE)*sqrt(0.5)*(randn(E,1) + 1i*randn(E,1));
%     while norm(h_UE_EVE - hest_UE_EVE)/norm(hest_UE_EVE) > channelUncertainty
%         hest_UE_EVE = h_UE_EVE ...
%         + channelUncertainty*norm(h_UE_EVE)*sqrt(0.5)*(randn(E,1) + 1i*randn(E,1));
%     end

    %% Some functions
    % Effective channel vector
    channel = @(alpha) h_UE_AP + H_EN_HRRIS*Upsilon(alpha)*h_UE_HRRIS;
    % Equivalent noise covariance
    Q = @(alpha) eye(M) + H_EN_HRRIS*Psi(alpha)*Psi(alpha)'*H_EN_HRRIS';
    % SINR
    SINR = @(w,alpha) real(P_UE*abs(w'*channel(alpha))^2./(noiseVar*w'*Q(alpha)*w));
    % Primary rate (UE --> EN)
    primaryRate = @(w,alpha) bandwidth*log2(1 + SINR(w,alpha));
    
    %% Optimization
    
    % Randomly initialize alpha
    alpha = randn(N,1) + 1i*randn(N,1);
    alpha = alpha./abs(alpha);
    alpha_randomPhase = alpha;
    
    % Compute the combining vector w accordingly
    w = sqrt(P_UE/noiseVar)*(Q(alpha)\channel(alpha));
    w_randomPhase = w;
    w_RIS = w;
    w_fixed = w;
    w_dynamic = w;
    
    % Precompute \xi (given after Eq. (20) in [1]
    xi = noiseVar + P_UE*abs(h_UE_HRRIS).^2;

    % Optimize w and alpha
    for idxIter = 1:nIterAO-1   % Alternating optimization
        %% Fix w, optimize the phases of alpha
        tmp1 = w_RIS'*h_UE_AP;
        tmp2 = diag(w_RIS'*H_EN_HRRIS)*h_UE_HRRIS;
        alpha_RIS = (tmp1/abs(tmp1))./(tmp2./abs(tmp2));

        tmp1 = w_fixed'*h_UE_AP;
        tmp2 = diag(w_fixed'*H_EN_HRRIS)*h_UE_HRRIS;
        alpha_fixed = (tmp1/abs(tmp1))./(tmp2./abs(tmp2));

        tmp1_dynamic = w_dynamic'*h_UE_AP;
        tmp2 = diag(w_dynamic'*H_EN_HRRIS)*h_UE_HRRIS;
        alpha_dynamic = (tmp1/abs(tmp1))./(tmp2./abs(tmp2));

        if N_relay > 0
            %% Fix w, optimize the amplitude of the active elements in fixed HRRIS
            for idxIterAlpha = 1:min(N_relay,5) % optimize each elements while keeping others fixed
                for n = Idx_relay(:)' % consider an active elements
                   % indices of all other active elements
                   idx_tmp = Idx_relay;
                   idx_tmp(idx_tmp == n) = [];

                   % Compute the coefficients given in [1, Eqs. (15)-(19)]
                   a = abs(h_UE_HRRIS).^2.*abs(H_EN_HRRIS'*w_fixed).^2;
                   GUhur = sum(abs(alpha_fixed.*h_UE_HRRIS.*(H_EN_HRRIS'*w_fixed))) ...
                       - abs(alpha_fixed.*h_UE_HRRIS.*(H_EN_HRRIS'*w_fixed));
                   tmp = (abs(w_fixed'*h_UE_AP) + GUhur);
                   tmp1 = abs(h_UE_HRRIS.*(H_EN_HRRIS'*w_fixed));
                   b = 2*tmp1.*tmp;
                   c = tmp.^2;
                   u = abs(H_EN_HRRIS'*w_fixed).^2;
                   v = w_fixed'*w_fixed + sum(tmp1(Idx_relay).^2) - tmp1.^2;

                   % Compute the term d given after [1, Eq. (22)]
                   d = v.*abs(h_UE_HRRIS).^2 - c;

                   % The term \tilde{P} given after [1, Eq. (20)]
                   P_tilde = (abs(alpha_fixed(idx_tmp)).^2).'*xi(idx_tmp);

%                    SINR_test = @(x) (P_UE/noiseVar)*(abs(h_UE_HRRIS(n))^2 + (b(n).*x - d(n))...
%                        ./(u(n).*x.^2 + v(n)));

                   % Optimize |alpha_n|, see [1, Eq. (21)]
                   % Option 1: empirical search
%                    Prt = linspace(0,(P_relay_max - P_tilde)/xi(n),1000);
%                    objtmp = zeros(size(Prt));
%                    for idxPtmp = 1:length(Prt)
%                        P_tmp = Prt(idxPtmp);
%                        alpha_tmp = alpha_fixed./abs(alpha_fixed);
%                        alpha_tmp(n) = alpha_tmp(n)*sqrt(P_tmp);
% 
%                        objtmp(idxPtmp) = SINR(w_fixed,alpha_tmp);
%                    end
%                    [~,idxPopt] = max(objtmp);
%                    ampopt = sqrt(Prt(idxPopt));
                   
                   % Option 2: closed form solution
                   ampopt = min(d(n)/b(n) + sqrt(d(n)^2/b(n)^2 + v(n)/u(n)),sqrt(Prt(end)));
                   
                   % test the closed form solution
%                    if idxPopt < length(Prt)
%                        sqrt(Prt(idxPopt))
%                        figure(1)
%                        plot(sqrt(Prt),objtmp)
%                        root = roots([-u(n)*b(n) 2*u(n)*d(n) v(n)*b(n)])
%                        d(n)/b(n) + sqrt(d(n)^2/b(n)^2 + v(n)/u(n))
%                        hold on
%                        plot(sqrt(Prt),SINR_test(sqrt(Prt)),'--')
%                        hold off
%                        keyboard
%                    end
                    alpha_fixed(n) = (alpha_fixed(n)/abs(alpha_fixed(n)))*ampopt;
                end
            end

            %% Fix w, optimize the position and amplitude of the active elements in dynamic HRRIS
            alpha_dynamic = alpha_dynamic./abs(alpha_dynamic);
            zeta = P*abs(h_UE_HRRIS).^2.*abs(H_EN_HRRIS'*w_dynamic).^2; %/2./abs(lambda);
            [~,idxActive] = maxk(zeta./xi,N_relay);

            % water filling
            wline = water_filling(xi(idxActive)./zeta(idxActive), ...
               P_relay_max, (P_relay_max/N_relay)*1e-6);
            alpha_dynamic(idxActive) = alpha_dynamic(idxActive).*...
               max(1,sqrt(max(wline./xi(idxActive) - 1./zeta(idxActive),0)));

            Idx_relay_dynamic = idxActive;  
            Idx_relay_dynamic_bin = zeros(N,1);  
            Idx_relay_dynamic_bin(Idx_relay_dynamic) = 1; 

            %% Fix alpha, optimize w
            Psi_dynamic = @(alpha) diag(alpha(:).*Idx_relay_dynamic_bin(:));
            Q_dynamic = @(alpha) eye(M) + ...
               H_EN_HRRIS*Psi_dynamic(alpha)*Psi_dynamic(alpha)'*H_EN_HRRIS';

            w_RIS = sqrt(P_UE/noiseVar)*channel(alpha_RIS);
            w_fixed = sqrt(P_UE/noiseVar)*(Q(alpha_fixed)\channel(alpha_fixed));
            w_dynamic = sqrt(P_UE/noiseVar)*(Q_dynamic(alpha_dynamic)\channel(alpha_dynamic));
        else
            w_RIS = sqrt(P_UE/noiseVar)*channel(alpha_RIS);
            w_fixed = w_RIS;
            w_dynamic = w_RIS;
        end
    end
    
    %% Fix w and alpha, optimize the offloading volume ell
    latency_randomPhase_tmp0 = zeros(length(fe_set),length(fl_set));
    latency_RIS_tmp0 = zeros(length(fe_set),length(fl_set));
    latency_fixed_tmp0 = zeros(length(fe_set),length(fl_set));
    latency_dynamic_tmp0 = zeros(length(fe_set),length(fl_set));
    
    for idxfl = 1:length(fl_set)
    fl = fl_set(idxfl);
    for idxfe = 1:length(fe_set)
    fe = fe_set(idxfe);
    nIterChannelEVE = 2*nIterMC;
    for idxhE = 1:nIterChannelEVE
        h_UE_EVE = channel_UE_EVE(:,:,idxhE);
        hest_UE_EVE = channel_UE_EVE_est(:,:,idxhE);
        
        leakageRate = bandwidth*log2(1+P_UE*norm(h_UE_EVE)^2/noiseVar);
        leakageRate_est = bandwidth*log2(1+P_UE*(1+channelUncertainty)^2*norm(hest_UE_EVE)^2/noiseVar);
        secrecyRate_est = @(w,alpha) max(0,primaryRate(w,alpha) - leakageRate_est);
        secrecyRate = @(w,alpha) max(0,primaryRate(w,alpha) - leakageRate);
        latency = @(w,alpha,ell) max((L-ell)*nu/fl,ell./secrecyRate(w,alpha) + ell*nu./fe);
    
        ell_randomPhase = L/(fl/nu/secrecyRate_est(w_randomPhase,alpha_randomPhase) + 1 + fl/fe);
        latency_randomPhase_tmp0(idxfe,idxfl) = latency_randomPhase_tmp0(idxfe,idxfl) + ...
            min(latency(w_randomPhase,alpha_randomPhase,ceil(ell_randomPhase)),...
                latency(w_randomPhase,alpha_randomPhase,floor(ell_randomPhase)))/nIterChannelEVE;
        
        ell_RIS = L/(fl/nu/secrecyRate_est(w_RIS,alpha_RIS) + 1 + fl/fe);
        latency_RIS_tmp0(idxfe,idxfl) = latency_RIS_tmp0(idxfe,idxfl) + ...
            min(latency(w_RIS,alpha_RIS,ceil(ell_RIS)),...
                latency(w_RIS,alpha_RIS,floor(ell_RIS)))/nIterChannelEVE;
        
        ell_fixed = L/(fl/nu/secrecyRate_est(w_fixed,alpha_fixed) + 1 + fl/fe);
        latency_fixed_tmp0(idxfe,idxfl) = latency_fixed_tmp0(idxfe,idxfl) + ...
            min(latency(w_fixed,alpha_fixed,ceil(ell_fixed)),...
                latency(w_fixed,alpha_fixed,floor(ell_fixed)))/nIterChannelEVE;
        
        ell_dynamic = L/(fl/nu/secrecyRate_est(w_dynamic,alpha_dynamic) + 1 + fl/fe);
        latency_dynamic_tmp0(idxfe,idxfl) = latency_dynamic_tmp0(idxfe,idxfl) + ...
            min(latency(w_dynamic,alpha_dynamic,ceil(ell_dynamic)),...
                latency(w_dynamic,alpha_dynamic,floor(ell_dynamic)))/nIterChannelEVE;
    end
    end
    end
    
    % Average the latency over the channel realizations
    latency_randomPhase_tmp = latency_randomPhase_tmp + latency_randomPhase_tmp0/nIterMC;
    latency_RIS_tmp = latency_RIS_tmp + latency_RIS_tmp0/nIterMC;
    latency_fixed_tmp = latency_fixed_tmp + latency_fixed_tmp0/nIterMC;
    latency_dynamic_tmp = latency_dynamic_tmp + latency_dynamic_tmp0/nIterMC;

    primRate_randomPhase_tmp = primRate_randomPhase_tmp + ...
            primaryRate(w_randomPhase,alpha_randomPhase)/nIterMC;
    primRate_RIS_tmp = primRate_RIS_tmp + primaryRate(w_RIS,alpha_RIS)/nIterMC;
    primRate_fixed_tmp = primRate_fixed_tmp + primaryRate(w_fixed,alpha_fixed)/nIterMC;
    primRate_dynamic_tmp = primRate_dynamic_tmp + primaryRate(w_dynamic,alpha_dynamic)/nIterMC;
end

% Collect the minimized latency and corresponding primary rate
latency_randomPhase(idxN,idxNrelay,idxP,idxdUE,idxdEVE,idxE,:,:) = latency_randomPhase_tmp;
latency_RIS(idxN,idxNrelay,idxP,idxdUE,idxdEVE,idxE,:,:) = latency_RIS_tmp;
latency_fixed(idxN,idxNrelay,idxP,idxdUE,idxdEVE,idxE,:,:) = latency_fixed_tmp;
latency_dynamic(idxN,idxNrelay,idxP,idxdUE,idxdEVE,idxE,:,:) = latency_dynamic_tmp;

primRate_randomPhase(idxN,idxNrelay,idxP,idxdUE,idxE,idxdEVE) = primRate_randomPhase_tmp;
primRate_RIS(idxN,idxNrelay,idxP,idxdUE,idxE,idxdEVE) = primRate_RIS_tmp;
primRate_fixed(idxN,idxNrelay,idxP,idxdUE,idxE,idxdEVE) = primRate_fixed_tmp;
primRate_dynamic(idxN,idxNrelay,idxP,idxdUE,idxE,idxdEVE) = primRate_dynamic_tmp;
end
end
end
end
end
end

% Remove singleton dimensions
latency_randomPhase = squeeze(latency_randomPhase);
latency_RIS = squeeze(latency_RIS);
latency_fixed = squeeze(latency_fixed);
latency_dynamic = squeeze(latency_dynamic);

primRate_randomPhase = squeeze(primRate_randomPhase);
primRate_RIS = squeeze(primRate_RIS);
primRate_fixed = squeeze(primRate_fixed);
primRate_dynamic = squeeze(primRate_dynamic);