#pragma once

#include <string>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <stdint.h>

// Thread-safe queue, blocking-wait
template <typename T>
class fifo_queue
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
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_queue.empty();
    }
    void push(T item)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_queue.push(item);
        }
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
    fifo_queue &operator=(const fifo_queue &rhs) // copy assignment
    {
        if (this == &rhs)
        {
            return *this;
        }
    }
};

typedef void *mxutil_stream_player_h;

/**
 * @brief Initialization of stream player also set the output display
 * frame resolution, which will be used in mxutil_stream_player_get_frame
 */
mxutil_stream_player_h mxutil_stream_player_open(const char *stream_url, const int disp_width, const int disp_height);
void mxutil_stream_player_close(mxutil_stream_player_h stream_handle);
void *mxutil_stream_player_get_frame(mxutil_stream_player_h stream_handle);
void mxutil_stream_player_return_buf(mxutil_stream_player_h stream_handle);
void mxutil_stream_get_input_resolution(mxutil_stream_player_h stream_handle, int &width, int &height);
std::string mxutil_stream_player_get_source_ip_addr(mxutil_stream_player_h stream_handle);
