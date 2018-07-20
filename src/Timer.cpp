#include "Timer.hpp"

#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <iomanip>

namespace RAILS
{

std::vector<Profile> profiles;

static Profile *find_profile(std::string const &name)
{
    std::vector<Profile>::iterator it = std::find(profiles.begin(), profiles.end(), name);
    return (it != profiles.end() ? &(*it) : NULL);
}

void timer_start(std::string const &name)
{
    Profile *current_profile = find_profile(name);

    if (current_profile == NULL)
    {
        Profile profile;
        profile.name = name;
        profiles.push_back(profile);
        current_profile = &profiles.back();
    }
    current_profile->calls++;
    current_profile->timer.ResetStartTime();
}

void timer_start(std::string const &cls, std::string const &name)
{
    timer_start(cls + "::" + name);
}

void timer_end(std::string const &name)
{
    Profile *current_profile = find_profile(name);
    if (current_profile != NULL)
        current_profile->time += current_profile->timer.ElapsedTime();
    else
        std::cerr << "Profile " << name << "not found\n";
}

void timer_end(std::string const &cls, std::string const &name)
{
    timer_end(cls + "::" + name);
}

void save_profiles(std::string const &filename)
{
    bool split_classes = true;
    if (!split_classes)
    {
        std::cout << std::setw(32) << "Name"
                  << std::setw(20) << "Total time"
                  << std::setw(20) << "Time per call"
                  << std::setw(10) << "Calls" << '\n';
        
        for (auto it = profiles.begin(); it != profiles.end(); ++it)
        {
            std::cout << std::setw(32) << it->name
                      << std::setw(20) << it->time
                      << std::setw(20) << it->time / (double)it->calls
                      << std::setw(10) << it->calls << '\n';
        }
    }
    else
    {
        std::cout << std::setw(32) << "Class"
                  << std::setw(20) << "Name"
                  << std::setw(20) << "Total time"
                  << std::setw(20) << "Time per call"
                  << std::setw(10) << "Calls" << '\n';
        
        for (auto it = profiles.begin(); it != profiles.end(); ++it)
        {
            std::string cls;
            std::string name;
            name = it->name;
            size_t pos = 0;
            pos = name.find("::");
            if (pos != std::string::npos)
            {
                cls = name.substr(0, pos);
                name.erase(0, pos + 2);
            }
            std::cout << std::setw(32) << cls
                      << std::setw(20) << name
                      << std::setw(20) << it->time
                      << std::setw(20) << it->time / (double)it->calls
                      << std::setw(10) << it->calls << '\n';
        }
    }
}
    
DestructibleTimer::DestructibleTimer(std::string const &_name)
    :
    name(_name)
{
    timer_start(name);
}

DestructibleTimer::DestructibleTimer(std::string const &cls, std::string const &_name)
    :
    name(cls+"::"+_name)
{
    timer_start(name);
}

DestructibleTimer::~DestructibleTimer()
{
    timer_end(name);
}

}
