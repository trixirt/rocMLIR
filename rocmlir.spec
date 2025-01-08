# Upstream tags are based on rocm releases:
%global rocm_release 6.3
%global rocm_patch 1
%global rocm_version %{rocm_release}.%{rocm_patch}

%global toolchain clang

%global upstreamname rocMLIR

%bcond_with debug
%if %{with debug}
%global build_type DEBUG
%else
%global build_type RelWithDebInfo
%endif

Name:           rocmlir
Version:        %{rocm_version}
Release:        1%{?dist}
Summary:        ROCm MLIR

Url:            https://github.com/ROCm/%{upstreamname}
# https://github.com/ROCm/rocMLIR/issues/1712
License:        Apache-2.0 AND Apache-2.0 WITH LLVM-exception OR NCSA
Source0:        %{url}/archive/rocm-%{rocm_version}.tar.gz#/%{name}-%{rocm_version}.tar.gz

BuildRequires:  binutils-devel
BuildRequires:  chrpath
BuildRequires:  cmake
BuildRequires:  gcc-c++
BuildRequires:  git
BuildRequires:  libffi-devel
BuildRequires:  libzstd-devel
BuildRequires:  perl
BuildRequires:  rocm-cmake
BuildRequires:  rocm-compilersupport-macros
BuildRequires:  rocm-device-libs
BuildRequires:  rocm-hip-devel
BuildRequires:  rocminfo
BuildRequires:  zlib-devel

# A fork of the rocm-llvm's code base, which commit is difficult to determine.
# The fork is built statically and not delivered.
# https://github.com/ROCm/rocMLIR/issues/1711
# NCSA and MIT
Provides:       bundled(rocm-llvm) = 20

%if 0%{?fedora} || 0%{?suse_version}
BuildRequires:  fdupes
%endif

ExclusiveArch:  x86_64

%description
This is the repository for a MLIR-based convolution and GEMM kernel
generator targetting AMD hardware. This generator is mainly used
from MIGraphX, but it can be used on a standalone basis. (The ability
to use this code via torch-mlir is being investigated as well.)

%package devel
Summary:        Libraries and headers for %{name}
Requires:       %{name}%{?_isa} = %{version}-%{release}

%description devel
%{summary}

%prep
%autosetup -p1 -n %{upstreamname}-rocm-%{rocm_version}

%build

# Maybe a circular dependency
# -DBUILD_FAT_LIBROCKCOMPILER=ON
    
%cmake \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=%{build_type} \
    -DCMAKE_CXX_COMPILER=%{rocmllvm_bindir}/clang++ \
    -DCMAKE_C_COMPILER=%{rocmllvm_bindir}/clang \
    -DCMAKE_SKIP_INSTALL_RPATH=TRUE \
    -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU" \
    -DLLVM_LIBDIR_SUFFIX=64 \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DMLIR_ENABLE_ROCM_RUNNER=OFF \
    -DROCM_PATH=%{_prefix}

%cmake_build

%install

%cmake_install

#Clean up dupes:
%if 0%{?fedora} || 0%{?suse_version}
%fdupes %{buildroot}%{_prefix}
%endif

# Remove rpaths that cmake missed
chrpath -d %{buildroot}%{_bindir}/*

# Remove unneeded files
rm %{buildroot}/usr/lib64/libconv*
rm -rf %{buildroot}/usr/lib64/objects-*

%files
%license mlir/LICENSE.TXT
%{_bindir}/%{name}-*
%{_libdir}/libGpuModule*.so.*
%{_libdir}/libMLIR*.so.*
%{_libdir}/libRocmlir*.so.*
%{_libdir}/libRocMLIR*.so.*


%files devel
%doc README.md
%{_libdir}/libGpuModule*.so
%{_libdir}/libMLIR*.so
%{_libdir}/libRocmlir*.so
%{_libdir}/libRocMLIR*.so

%changelog
* Wed Jan 8 2025 Tom Rix <Tom.Rix@amd.com> - 6.3.1-1
- Initial version
