#ifndef LYAPUNOV_TIMER_H
#define LYAPUNOV_TIMER_H

#include <mpi.h>
#include <ctime>
#include <string>

class Timer
{
double startTime_;
public:
    Timer()
        :
        startTime_(0.0)
        {}
        
    double wallTime()
        {
            int mpiInit;
            MPI_Initialized(&mpiInit);
            if (mpiInit)
                return MPI_Wtime();
            else
                return (double) std::clock() / CLOCKS_PER_SEC;
        }
        
    void ResetStartTime()
        { startTime_ = wallTime();}
        
    double ElapsedTime()
        {
            if (startTime_ > 0.0)
                return (double) (wallTime() - startTime_);
            else
                return 0.0;
        }
};

class Profile
{
public:
    std::string name;
    int calls;
    Timer timer;
    double time;

    Profile()
        :
        calls(0),
        timer(),
        time(0.0)
        {}

    bool operator ==(std::string const &other)
        {
            return !(name.compare(other));
        }
};

void timer_start(std::string const &name);
void timer_start(std::string const &cls, std::string const &name);
void timer_end(std::string const &name);
void timer_end(std::string const &cls, std::string const &name);
void save_profiles(std::string const &filename);

class DestructibleTimer
{
    std::string name;

public:
    DestructibleTimer(std::string const &name);
    DestructibleTimer(std::string const &cls, std::string const &name);
    ~DestructibleTimer();
};

#ifndef TIMER_ON

#define TIMER_START(args...)
#define TIMER_END(args...)
#define START_TIMER(args...)
#define END_TIMER(args...)
#define FUNCTION_TIMER(args...)
#define SAVE_PROFILES(args...)

#else

#define TIMER_START(args...) timer_start(args)
#define TIMER_END(args...) timer_end(args)
#define START_TIMER(args...) timer_start(args)
#define END_TIMER(args...) timer_end(args)
#define FUNCTION_TIMER(args...) DestructibleTimer function_timer(args)
#define SAVE_PROFILES(args...) save_profiles(args)

#endif

#endif
