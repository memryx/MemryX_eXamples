#pragma once

#include <mutex>
#include <queue>
#include <condition_variable>

// Thread-safe queue
template <typename T>
class mxutil_fifo_queue
{
private:
    std::queue<T> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cond;

public:
    size_t size()
    {
        return m_queue.size();
    }

    bool empty()
    {
        return m_queue.empty();
    }
    void push(T item)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_queue.push(item);
        m_cond.notify_one();
    }
    T pop()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cond.wait(lock,
                    [this]()
                    { return !m_queue.empty(); });
        T item = m_queue.front();
        m_queue.pop();
        return item;
    }
};