#ifndef TAPA_HOST_BUFFER_H_
#define TAPA_HOST_BUFFER_H_

#include "tapa/base/buffer.h"
#include "tapa/host/stream.h"

namespace tapa {

template <typename T, int n_sections, typename... dims>
class buffer;

template <typename T, int n_sections, typename... dims>
class ibuffer;

template <typename T, int n_sections, typename... dims>
class obuffer;

template <typename T, int n_sections, typename... dims>
class section;

template <typename T, int len, int n_sections, typename... dims>
class buffers;

template <typename T, int len, int n_sections, typename... dims>
class ibuffers;

template <typename T, int len, int n_sections, typename... dims>
class obuffers;

namespace internal {

template <typename T, int len, int n_sections, typename... dims>
class basic_buffers;

template <typename T, int n_sections>
struct buffer_data {
  buffer_data(const std::string& name = "")
      : name(name), ptr(new T[n_sections]) {
    for (int i = 0; i < n_sections; i++) {
      free_sections.write(i);
    }
    free_sections.set_name(name + "'s free sections FIFO");
    occupied_sections.set_name(name + "'s occupied sections FIFO");
  }

  ~buffer_data() {
    // to avoid the warning that filled FIFOs were destroyed
    for (int i = 0; i < n_sections; i++) {
      free_sections.read();
    }
  }

  const std::string& get_name() const { return this->name; }
  void set_name(const std::string& name) {
    this->name = name;
    free_sections.set_name(this->name + "'s free sections FIFO");
    occupied_sections.set_name(this->name + "'s occupied sections FIFO");
  }

  stream<int, n_sections> free_sections;
  stream<int, n_sections> occupied_sections;
  // the memory buffer is an std::array wrapped by std::shared_ptr because
  // while being passed down to the task, the buffer object gets copied
  // because of std::forward; it should be a single unique buffer in all
  // copies and should not be freed when these copies go out of scope;
  // std::shared_ptr insures this
  std::shared_ptr<T[n_sections]> ptr;
  std::string name;
};

template <typename T, int n_sections>
class basic_buffer {
 protected:
  basic_buffer(const std::shared_ptr<buffer_data<T, n_sections>>& ptr)
      : inner_data(ptr) {}

  basic_buffer(const basic_buffer&) = default;
  basic_buffer(basic_buffer&&) = default;
  basic_buffer& operator=(const basic_buffer&) = default;
  basic_buffer& operator=(basic_buffer&&) = delete;  // -Wvirtual-move-assign

  const std::string& get_name() const { return inner_data->get_name(); }
  void set_name(const std::string& name) { inner_data->set_name(name); }

 public:
  std::shared_ptr<buffer_data<T, n_sections>> inner_data;
};

}  // namespace internal

/// Provides producer-side access to a @c tapa::buffer object where it is used
/// as an @a input
///
/// This class should only be used in task function parameters and should never
/// be instatiated directly.
//
/// Example usage as follows.
/// ```cpp
/// void producer_task(tapa::obuffer& buf_out, ...) {
///   {
///     // acquire a free section of buf_out
///     auto buf_out_accessor = buf_out.acquire();
///     // read/write the section like a regular array
///     for(int i = 0; i < BUF_SIZE; i++) {
///       buf_out_accessor[i] = ...;
///     }
///     // as buf_out_accessor goes out of scope, the
///     // section is marked as occupied so the
///     // consumer side can read it
///   }
/// }
/// ```
template <typename T, int n_sections, typename... dims>
class obuffer : public virtual internal::basic_buffer<T, n_sections> {
 protected:
  obuffer() : internal::basic_buffer<T, n_sections>(nullptr) {}

 public:
  using section_t = section<T, n_sections, dims...>;
  section_t acquire() { return section_t(*this, true); }
};

/// Provides consumer-side access to a @c tapa::buffer object where it is used
/// as an @a input
///
/// This class should only be used in task function parameters and should never
/// be instatiated directly.
///
/// Example usage as follows.
/// ```cpp
/// void consumer_task(tapa::ibuffer& buf_in, ...) {
///   {
///     // acquire an occupied section of buf_in
///     auto buf_in_accessor = buf_in.acquire();
///     // read/write the section like a regular array
///     for(int i = 0; i < BUF_SIZE; i++) {
///       buf_in_accessor[i] = ...;
///     }
///     // as buf_in_accessor goes in of scope, the
///     // section is marked as free so the producer
///     // side can write to it again
///   }
/// }
/// ```
template <typename T, int n_sections, typename... dims>
class ibuffer : public virtual internal::basic_buffer<T, n_sections> {
 protected:
  ibuffer() : internal::basic_buffer<T, n_sections>(nullptr) {}

 public:
  using section_t = section<T, n_sections, dims...>;
  section_t acquire() { return section_t(*this, false); }
};

// diamond inheritance so that `buffer&` can be cast to `ibuffer&` and
// `obuffer&` but they can't be cast to each other

/// Defines a PingPong Buffer based communication channel between
/// two task instances.
///
/// To use the PingPong buffer, instantiate `tapa::buffer` in parent
/// task and pass it either as `tapa::ibuffer` or `tapa::obuffer`
/// reference to children tasks.
///
/// Internally, the buffer object keeps multiple sections, indicated
/// by the template parameter @c n_sections. A section is said to be
/// free if it can be written to by the producer task and it is said to
/// be occupied if it can be read from by the consumer task.
///
/// @tparam T the data type to store in the buffer
/// @tparam n_sections the total number of PingPong buffers; mostly two.
template <typename T, int n_sections, typename... dims>
class buffer : public ibuffer<T, n_sections, dims...>,
               public obuffer<T, n_sections, dims...> {
 public:
  buffer(const std::string& name = "")
      : internal::basic_buffer<T, n_sections>(
            std::make_shared<internal::buffer_data<T, n_sections>>(name)) {}
};

// A helper class to let users access a section of
// the pingpong buffer memory core
template <typename T, int n_sections, typename... dims>
class section {
 public:
  T& operator()() { return data.inner_data->ptr[section_id]; }
  const T& operator()() const { return data.inner_data->ptr[section_id]; }

  ~section() {
    if (for_producer) {
      data.inner_data->occupied_sections.write(section_id);
    } else {
      data.inner_data->free_sections.write(section_id);
    }
  }

 private:
  using buffer_t = internal::basic_buffer<T, n_sections>;

  // section class should be instantiated only by an `ibuffer`
  // or `obuffer`; Hence private constructor and friend class
  // declarations
  friend class obuffer<T, n_sections, dims...>;
  friend class ibuffer<T, n_sections, dims...>;

  // block on the src fifo for the section ID
  section(buffer_t& buf, bool for_producer = true)
      : data(buf), for_producer(for_producer) {
    if (for_producer) {
      section_id = data.inner_data->free_sections.read();
    } else {
      section_id = data.inner_data->occupied_sections.read();
    }
  }

  // the actual buffer object and the section_id this instance
  // is supposed to access
  buffer_t& data;
  int section_id;
  // whether the instance is for a producer task or a consumer task
  const bool for_producer;
};

namespace internal {

template <typename T, int len, int n_sections, typename... dims>
class basic_buffers {
  // TODO: This shouldn't be public as `ibuffers` and `obuffers`
  // are both friend classes, for some reason, g++ compiles that fine
  // however we run into trouble with Clang. Before I get to the bottom
  // of this, I'm making metadata_t public. This should be fixed!
 public:
  using buffer_t = buffer<T, n_sections, dims...>;

  struct metadata_t {
    metadata_t(const std::string& name, int pos) : name(name), pos(pos) {}
    const std::string name;
    const int pos;
    std::vector<buffer_t> refs;
  };

 protected:
  template <typename f_T, int f_len, int f_n_sections, typename... f_dims>
  friend class ibuffers;

  template <typename f_T, int f_len, int f_n_sections, typename... f_dims>
  friend class obuffers;

  template <typename f_T, int f_len, int f_n_sections, typename... f_dims>
  friend class buffers;

  basic_buffers(const std::shared_ptr<metadata_t>& ptr) : ptr(ptr) {}
  basic_buffers(const basic_buffers&) = default;
  basic_buffers(basic_buffers&&) = default;
  basic_buffers& operator=(const basic_buffers&) = default;
  basic_buffers& operator=(basic_buffers&&) = delete;  // -Wvirtual-move-assign

  const std::string& get_name() const { return ptr->name; }
  void set_name(const std::string& name) { ptr->name = name; }

  buffer_t operator[](int pos) const {
    CHECK_NOTNULL(ptr.get());
    CHECK_GE(pos, 0);
    CHECK_LT(pos, ptr->refs.size());
    return ptr->refs[pos];
  }

  std::shared_ptr<metadata_t> ptr;
};

}  // namespace internal

template <typename T, int len, int n_sections, typename... dims>
class ibuffers
    : public virtual internal::basic_buffers<T, len, n_sections, dims...> {
  using ibuffer_t = ibuffer<T, n_sections, dims...>;
  using buffer_t = buffer<T, n_sections, dims...>;
  using basic_buffers_t = internal::basic_buffers<T, len, n_sections, dims...>;

  template <int length>
  using ibuffers_slice_t = ibuffers<T, length, n_sections, dims...>;

  template <int length>
  using basic_buffers_slice_t =
      internal::basic_buffers<T, length, n_sections, dims...>;

 public:
  ibuffer_t& operator[](int pos) const { return this->ptr->refs[pos]; }

 protected:
  template <typename Param, typename Arg>
  friend struct internal::accessor;

  template <typename f_T, int f_len, int f_n_sections, typename... f_dims>
  friend class ibuffers;

  int access_pos_ = 0;

  ibuffers() : basic_buffers_t(nullptr) {}

  buffer<T, n_sections, dims...> access_() {
    CHECK_LT(access_pos_, this->ptr->refs.size())
        << "channels '" << this->ptr->name << "' accessed as ibuffer for "
        << access_pos_ + 1 << " times but it only contains "
        << this->ptr->refs.size() << " channels";
    return basic_buffers_t::operator[](access_pos_++);
  }

  template <int length>
  ibuffers_slice_t<length> access() {
    CHECK_NOTNULL(this->ptr.get());
    ibuffers_slice_t<length> result;
    result.ptr =
        std::make_shared<typename basic_buffers_slice_t<length>::metadata_t>(
            this->ptr->name, access_pos_);
    result.ptr->refs.reserve(length);
    for (int i = 0; i < length; i++) {
      result.ptr->refs.emplace_back(access_());
    }
    return result;
  }

  ibuffer_t access() { return access_(); }
};

template <typename T, int len, int n_sections, typename... dims>
class obuffers
    : public virtual internal::basic_buffers<T, len, n_sections, dims...> {
  using obuffer_t = obuffer<T, n_sections, dims...>;
  using buffer_t = buffer<T, n_sections, dims...>;
  using basic_buffers_t = internal::basic_buffers<T, len, n_sections, dims...>;

  template <int length>
  using obuffers_slice_t = obuffers<T, length, n_sections, dims...>;

  template <int length>
  using basic_buffers_slice_t =
      internal::basic_buffers<T, length, n_sections, dims...>;

 public:
  obuffer_t& operator[](int pos) const { return this->ptr->refs[pos]; }

 protected:
  template <typename Param, typename Arg>
  friend struct internal::accessor;

  template <typename f_T, int f_len, int f_n_sections, typename... f_dims>
  friend class obuffers;

  int access_pos_ = 0;

  obuffers() : basic_buffers_t(nullptr) {}

  buffer_t access_() {
    CHECK_LT(access_pos_, this->ptr->refs.size())
        << "channels '" << this->ptr->name << "' accessed as obuffer for "
        << access_pos_ + 1 << " times but it only contains "
        << this->ptr->refs.size() << " channels";
    return basic_buffers_t::operator[](access_pos_++);
  }

  template <int length>
  obuffers_slice_t<length> access() {
    CHECK_NOTNULL(this->ptr.get());
    obuffers_slice_t<length> result;
    result.ptr =
        std::make_shared<typename basic_buffers_slice_t<length>::metadata_t>(
            this->ptr->name, access_pos_);
    result.ptr->refs.reserve(length);
    for (int i = 0; i < length; i++) {
      result.ptr->refs.emplace_back(access_());
    }
    return result;
  }

  obuffer_t access() { return access_(); }
};

template <typename T, int len, int n_sections, typename... dims>
class buffers : public ibuffers<T, len, n_sections, dims...>,
                public obuffers<T, len, n_sections, dims...> {
  using basic_buffers_t = internal::basic_buffers<T, len, n_sections, dims...>;
  using ibuffer_t = ibuffer<T, n_sections, dims...>;
  using obuffer_t = obuffer<T, n_sections, dims...>;
  using buffer_t = buffer<T, n_sections, dims...>;

  using ibuffers_t = ibuffers<T, len, n_sections, dims...>;
  using obuffers_t = obuffers<T, len, n_sections, dims...>;

  template <int length>
  using ibuffers_slice_t = ibuffers<T, length, n_sections, dims...>;

  template <int length>
  using obuffers_slice_t = obuffers<T, length, n_sections, dims...>;

 public:
  buffers(const std::string& name = "")
      : basic_buffers_t(
            std::make_shared<typename basic_buffers_t::metadata_t>(name, 0)) {
    this->ptr->refs.reserve(len);
    for (int i = 0; i < len; i++) {
      this->ptr->refs.emplace_back(name + ":" + std::to_string(i));
    }
  }

  buffer_t operator[](int pos) const {
    return basic_buffers_t::operator[](pos);
  }

 protected:
  template <typename Param, typename Arg>
  friend struct internal::accessor;

  obuffer_t access_as_obuffer() { return obuffers_t::access(); }

  template <int length>
  obuffers_slice_t<length> access_as_obuffers() {
    return obuffers_t::template access<length>();
  }

  template <int length>
  ibuffers_slice_t<length> access_as_ibuffers() {
    return ibuffers_t::template access<length>();
  }

  ibuffer_t access_as_ibuffer() { return ibuffers_t::access(); }
};

namespace internal {

#define TAPA_DEFINE_ACCESSER(io, reference)                                \
  /* param = i/obuffer, arg = buffers */                                   \
  template <typename T, int len, int n_sections, typename... dims>         \
  struct accessor<io##buffer<T, n_sections, dims...> reference,            \
                  buffers<T, len, n_sections, dims...>&> {                 \
    static io##buffer<T, n_sections, dims...> access(                      \
        buffers<T, len, n_sections, dims...>& arg) {                       \
      return arg.access_as_##io##buffer();                                 \
    }                                                                      \
  };                                                                       \
                                                                           \
  /* param = i/obuffer, arg = i/obuffers */                                \
  template <typename T, int len, int n_sections, typename... dims>         \
  struct accessor<io##buffer<T, n_sections, dims...> reference,            \
                  io##buffers<T, len, n_sections, dims...>&> {             \
    static io##buffer<T, n_sections, dims...> access(                      \
        io##buffers<T, len, n_sections, dims...>& arg) {                   \
      return arg.access();                                                 \
    }                                                                      \
  };                                                                       \
                                                                           \
  /* param = i/obuffers, arg = buffers */                                  \
  template <typename T, int len, int n_sections, int left_len,             \
            typename... dims>                                              \
  struct accessor<io##buffers<T, left_len, n_sections, dims...> reference, \
                  buffers<T, len, n_sections, dims...>&> {                 \
    static io##buffers<T, left_len, n_sections, dims...> access(           \
        buffers<T, len, n_sections, dims...>& arg) {                       \
      return arg.template access_as_##io##buffers<left_len>();             \
    }                                                                      \
  };                                                                       \
                                                                           \
  /* param = i/obuffers, arg = i/obuffers */                               \
  template <typename T, int len, int n_sections, int left_len,             \
            typename... dims>                                              \
  struct accessor<io##buffers<T, left_len, n_sections, dims...> reference, \
                  io##buffers<T, len, n_sections, dims...>&> {             \
    static io##buffers<T, left_len, n_sections, dims...> access(           \
        io##buffers<T, len, n_sections, dims...>& arg) {                   \
      return arg.template access<left_len>();                              \
    }                                                                      \
  };

TAPA_DEFINE_ACCESSER(i, )
TAPA_DEFINE_ACCESSER(i, &)
TAPA_DEFINE_ACCESSER(i, &&)
TAPA_DEFINE_ACCESSER(o, )
TAPA_DEFINE_ACCESSER(o, &)
TAPA_DEFINE_ACCESSER(o, &&)

#undef TAPA_DEFINE_ACCESSER
}  // namespace internal

}  // namespace tapa

#endif  // TAPA_HOST_BUFFER_H_