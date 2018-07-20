#ifndef RAILS_LYAPUNOV_TIMER_H
#define RAILS_LYAPUNOV_TIMER_H

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <ctime>
#include <string>

namespace RAILS
{

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
#ifdef HAVE_MPI
            int mpiInit;
            MPI_Initialized(&mpiInit);
            if (mpiInit)
                return MPI_Wtime();
            else
#endif
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

}

#ifndef TIMER_ON

#define RAILS_TIMER_START(args...)
#define RAILS_TIMER_END(args...)
#define RAILS_START_TIMER(args...)
#define RAILS_END_TIMER(args...)
#define RAILS_FUNCTION_TIMER(args...)
#define RAILS_SAVE_PROFILES(args...)

#else

#define RAILS_TIMER_START(args...) timer_start(args)
#define RAILS_TIMER_END(args...) timer_end(args)
#define RAILS_START_TIMER(args...) timer_start(args)
#define RAILS_END_TIMER(args...) timer_end(args)
#define RAILS_FUNCTION_TIMER(args...) RAILS::DestructibleTimer function_timer(args)
#define RAILS_SAVE_PROFILES(args...) save_profiles(args)

#endif

#endif
