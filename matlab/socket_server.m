% SOCKET_SERVER - run a server that listens on a socket for computations to perform 

function socket_server(port,debug)
    
    if isdeployed
	port = str2num(port);
    end
    
    if nargin==2 && isdeployed
        debug = str2num(debug);
    end
    
    if nargin<2 || isempty(debug)
        debug = 0;
    end
    
    codes = messagecodes;
    
    socket = mslisten(port);
    
    if debug
        fprintf('Opened a socket on port %d with number %d\n',port,socket);
    end
    
    while 1
        % Keep listening until a connection is received
        sock = -1;
        while sock == -1
            sock = msaccept(socket,0.0000001);
            drawnow;
        end
        fprintf('Accepted a connection\n');
        % Send an acknowledgement
        m.accepted = 1;
        mssend(sock,m);
        
        while 1
            success = -1;
            clear rv;
            while success<0
                [received,success] = msrecv(sock,0.0000001);
                %WaitSecs(0.000001);
                drawnow;
            end
            switch received.command
              case {codes.dummy}
                % do nothing
                if debug
                    fprintf('Received dummy command\n');
                end
              case {codes.decompose}
                if debug
                    fprintf('Received decompose command\n');
                end
                [time,vel,numsubmovements,method,algorithm] = deal(received.arguments{:});
                [rv.best,rv.bestresult,rv.bestfitresult] = decompose(time,vel,numsubmovements,method,algorithm);
                mssend(sock,rv);
                
              case {codes.closesocket}
                msclose(sock);
                if debug
                    fprintf('Closed socket\n');
                end
                break;
            end
        end
    end
