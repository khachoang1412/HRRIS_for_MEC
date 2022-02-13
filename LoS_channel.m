function H = LoS_channel(N, Nt, Nr, channel)
Nx = 5; % assume that 5 elements of the horizontal dimension of RIS

if channel == 1  % BS-RIS channel --------------------------------------------
    AoD = 2*pi * rand;
    AoA_az = 2*pi * rand;
    AoA_el = pi*rand - pi/2;
    
    % Tx array response vector
    a_tx = exp(1i * [0:Nt-1] .* pi * sin(AoD)).';
    
    % Rx array response vector
    sinsin = sin(AoA_az)*sin(AoA_el);
    sincos = sin(AoA_az)*cos(AoA_el);
    n = [0:N-1];
    a_rx = exp(1i * pi * ( floor(n./Nx).*sinsin + (n-floor(n./Nx)*Nx) * sincos )).';
    
    % LoS channel
    H = a_rx*a_tx';
else             % RIS-MS channel --------------------------------------------
    AoD = 2*pi * rand;
    AoA_az = 2*pi * rand;
    AoA_el = pi*rand - pi/2;
    
    % Rx array response vector
    a_rx = exp(1i * [0:Nr-1] .* pi * sin(AoD)).';
    
    % Tx array response vector
    sinsin = sin(AoA_az)*sin(AoA_el);
    sincos = sin(AoA_az)*cos(AoA_el);
    n = [0:N-1];
    a_tx = exp(1i * pi * ( floor(n./Nx).*sinsin + (n-floor(n./Nx)*Nx) * sincos )).';
    
    % LoS channel
    H = a_rx*a_tx';
end
end